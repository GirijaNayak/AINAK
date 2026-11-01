from __future__ import annotations
import csv, json, random, hashlib, platform, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from ..models.gesture.lstm import GestureLSTM,GESTURE_LABELS,FEATURES,SEQUENCE_LENGTH


class GestureSequenceDataset(Dataset):
    def __init__(self, manifest: Path, groups: set[str] | None = None):
        self.items=[]
        with manifest.open(encoding="utf-8-sig",newline="") as f:
            for row in csv.DictReader(f):
                if row.get("synthetic", "").lower() in {"1", "true", "yes"}:
                    raise ValueError("synthetic gesture sequences are prohibited as real-data training evidence")
                if row.get("reviewed", "").strip().lower() not in {"true", "yes", "1"}:
                    raise ValueError("gesture training requires reviewed human annotations; use extract-gesture on a reviewed session")
                if not row.get("annotator", "").strip() or not row.get("provenance", "").strip():
                    raise ValueError("gesture training rows must identify annotator and annotation provenance")
                annotation_confidence=float(row.get("annotation_confidence", "nan"))
                if not np.isfinite(annotation_confidence) or not 0 <= annotation_confidence <= 1:
                    raise ValueError("gesture annotation_confidence must be finite and in [0,1]")
                # Prefer participant-level splitting where participant codes exist;
                # otherwise split by the recording session to avoid adjacent-frame leakage.
                group=row.get("subject_id") or row.get("recording_id") or Path(row["path"]).stem
                if groups is not None and group not in groups: continue
                if row["label"] not in GESTURE_LABELS: raise ValueError(f"unsupported gesture label {row['label']}")
                self.items.append((np.load(manifest.parent/row["path"]).astype(np.float32),GESTURE_LABELS.index(row["label"]),group))
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        x,y,_=self.items[i]
        if x.ndim!=2 or x.shape[1]!=FEATURES: raise ValueError(f"expected sequence [frames,{FEATURES}]")
        if x.shape[0] < 2: raise ValueError("gesture sequence must contain at least two frames")
        # Resample each labeled clip to the model's fixed 24-frame time grid.
        positions=np.linspace(0,x.shape[0]-1,SEQUENCE_LENGTH)
        lo=np.floor(positions).astype(int);hi=np.minimum(lo+1,x.shape[0]-1);weight=(positions-lo)[:,None]
        x=(1-weight)*x[lo]+weight*x[hi]
        return torch.from_numpy(x.astype(np.float32)),torch.tensor(y,dtype=torch.long)


def train_gesture(manifest: Path, output: Path, epochs: int=30, batch_size: int=16, seed: int=17, device: str="cpu",
                  patience: int=5, hidden_size: int=128, layers: int=2, overwrite: bool=False):
    started=time.perf_counter();manifest_hash=hashlib.sha256(manifest.read_bytes()).hexdigest()
    if epochs<1 or batch_size<1 or patience<1:raise ValueError("epochs, batch_size, and patience must be positive")
    if output.exists() and not overwrite:raise FileExistsError(f"refusing to overwrite checkpoint: {output}; pass --overwrite explicitly")
    metrics_path=output.with_suffix(".metrics.json")
    if metrics_path.exists() and not overwrite:raise FileExistsError(f"refusing to overwrite training report: {metrics_path}; pass --overwrite explicitly")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    full=GestureSequenceDataset(manifest); group_ids=sorted({item[2] for item in full.items})
    if len(group_ids)<3: raise ValueError("gesture training requires at least three independent participants or sessions for train/validation/test separation")
    random.shuffle(group_ids); n_test=max(1,round(len(group_ids)*.2)); n_val=max(1,round(len(group_ids)*.2))
    if len(group_ids)-n_test-n_val<1:n_val=len(group_ids)-n_test-1
    test_ids=set(group_ids[:n_test]);val_ids=set(group_ids[n_test:n_test+n_val]);train_ids=set(group_ids[n_test+n_val:])
    train=GestureSequenceDataset(manifest,train_ids); val=GestureSequenceDataset(manifest,val_ids);test=GestureSequenceDataset(manifest,test_ids)
    if not len(train) or not len(val) or not len(test): raise ValueError("empty train, validation, or test split")
    tr=DataLoader(train,batch_size=batch_size,shuffle=True); va=DataLoader(val,batch_size=batch_size);te=DataLoader(test,batch_size=batch_size)
    train_counts=np.bincount([item[1] for item in train.items],minlength=len(GESTURE_LABELS))
    if np.any(train_counts==0):raise ValueError(f"training split must contain every gesture class; counts={dict(zip(GESTURE_LABELS,train_counts.tolist()))}")
    class_weights=len(train.items)/(len(GESTURE_LABELS)*train_counts.astype(np.float32))
    model_config={"hidden_size":hidden_size,"layers":layers}
    model=GestureLSTM(**model_config).to(device); opt=torch.optim.AdamW(model.parameters(),lr=1e-3)
    loss_fn=nn.CrossEntropyLoss(weight=torch.tensor(class_weights,dtype=torch.float32,device=device)); best=float("inf"); best_state=None;best_yt=[];best_yp=[]
    history=[]; stale_epochs=0
    for epoch in range(epochs):
        model.train()
        train_losses=[]
        for x,y in tr:
            x,y=x.to(device),y.to(device); logits,_=model(x); loss=loss_fn(logits,y); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step(); train_losses.append(float(loss.item()))
        model.eval(); losses=[]; yp=[]; yt=[]
        with torch.no_grad():
            for x,y in va:
                logits,_=model(x.to(device)); losses.append(float(loss_fn(logits,y.to(device)).item())); yp.extend(logits.argmax(-1).cpu().tolist()); yt.extend(y.tolist())
        score=float(np.mean(losses))
        history.append({"epoch":epoch+1,"training_loss":float(np.mean(train_losses)),"validation_loss":score})
        if score<best:
            best=score; best_state={k:v.cpu().clone() for k,v in model.state_dict().items()};best_yt=list(yt);best_yp=list(yp);stale_epochs=0
        else:
            stale_epochs+=1
            if stale_epochs>=patience:break
    model.load_state_dict(best_state);model.eval();test_yt=[];test_yp=[]
    with torch.no_grad():
        for x,y in te:
            logits,_=model(x.to(device));test_yp.extend(logits.argmax(-1).cpu().tolist());test_yt.extend(y.tolist())
    output.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"checkpoint_schema":"ainak.gesture-checkpoint.v1","state_dict":best_state,"model_config":model_config,"preprocessing":{"landmarks":33,"features_per_landmark":["x","y","z","visibility"],"center":"midpoint of hips 23/24","scale":"shoulder distance 11/12","low_visibility_threshold":.2,"sequence_length":SEQUENCE_LENGTH},"labels":GESTURE_LABELS,"seed":seed,"sequence_length":SEQUENCE_LENGTH,
        "manifest":str(manifest),"manifest_sha256":manifest_hash,"recording_split":{"train":sorted(train_ids),"validation":sorted(val_ids),"test":sorted(test_ids)},
        "train_groups":sorted(train_ids),"validation_groups":sorted(val_ids),"test_groups":sorted(test_ids),
        "training_examples":len(train),"validation_examples":len(val),"test_examples":len(test),
        "trained_on_real_data":True,"training_data_provenance":"explicitly reviewed human gesture annotations",
        "training_metadata":{"python":platform.python_version(),"torch":torch.__version__,"manifest_sha256":manifest_hash,"epochs_requested":epochs,"epochs_completed":len(history),"patience":patience,"batch_size":batch_size,"duration_seconds":time.perf_counter()-started}},output)
    labels=list(range(len(GESTURE_LABELS)))
    report=classification_report(best_yt,best_yp,labels=labels,target_names=GESTURE_LABELS,output_dict=True,zero_division=0)
    test_report=classification_report(test_yt,test_yp,labels=labels,target_names=GESTURE_LABELS,output_dict=True,zero_division=0)
    metrics={"validation_loss":best,"validation_classification_report":report,"validation_confusion_matrix":confusion_matrix(best_yt,best_yp,labels=labels).tolist(),
        "heldout_test":{"sequences":len(test),"classification_report":test_report,"confusion_matrix":confusion_matrix(test_yt,test_yp,labels=labels).tolist()},
        "train_sequences":len(train),"validation_sequences":len(val),"manifest_sha256":manifest_hash,"seed":seed,"history":history,
        "group_split":{"train":sorted(train_ids),"validation":sorted(val_ids),"test":sorted(test_ids)},
        "class_weights":dict(zip(GESTURE_LABELS,class_weights.tolist())),"duration_seconds":time.perf_counter()-started}
    metrics_path.write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    return metrics
