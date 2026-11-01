from __future__ import annotations
import json, random, hashlib, platform, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from ..data.validation import parse_example
from ..data.schemas import ALERT_TYPES
from ..models.fusion.transformer import MultimodalTransformer, MODALITIES, DEFAULT_FEATURE_DIMS


class FusionManifestDataset(Dataset):
    def __init__(self, path: Path, seq_len: int = 8, recording_ids: set[str] | None = None):
        grouped=defaultdict(list)
        group_for_recording={}
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                row=json.loads(line); ex=parse_example(row)
                rid=ex.recording_id or ex.example_id
                split_group=ex.participant_id or rid
                group_for_recording[rid]=split_group
                if recording_ids is None or split_group in recording_ids: grouped[rid].append((ex,row))
        self.items=[]; self.seq_len=seq_len
        self.split_groups=set()
        for recording, entries in grouped.items():
            entries.sort(key=lambda pair:pair[0].timestamp)
            self.split_groups.add(group_for_recording[recording])
            for end in range(len(entries)):
                window=entries[max(0,end-seq_len+1):end+1]
                self.items.append((recording,window))
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        _, window=self.items[idx]; t=len(window)
        features={m:torch.zeros(self.seq_len,DEFAULT_FEATURE_DIMS[m]) for m in MODALITIES}
        mask=torch.zeros(self.seq_len,3,dtype=torch.bool); times=torch.zeros(self.seq_len)
        start=window[0][0].timestamp
        for i,(ex,row) in enumerate(window, self.seq_len-t):
            times[i]=float(ex.timestamp-start)
            for j,m in enumerate(MODALITIES):
                obs=getattr(ex,m)
                if obs is not None:
                    vec=torch.tensor(obs.feature,dtype=torch.float32)
                    if vec.numel()!=DEFAULT_FEATURE_DIMS[m]: raise ValueError(f"{m} feature dimension {vec.numel()} != {DEFAULT_FEATURE_DIMS[m]}")
                    features[m][i]=vec; mask[i,j]=True
        priority=torch.tensor(window[-1][0].priority,dtype=torch.float32)
        confidence=torch.tensor(window[-1][0].confidence,dtype=torch.float32)
        alert=torch.tensor(ALERT_TYPES.index(window[-1][0].alert_type),dtype=torch.long)
        return features,mask,times,priority,confidence,alert


def collate(batch):
    fs,ms,ts,ps,cs,as_ = zip(*batch)
    return {m:torch.stack([f[m] for f in fs]) for m in MODALITIES},torch.stack(ms),torch.stack(ts),torch.stack(ps),torch.stack(cs),torch.stack(as_)


def train_fusion(manifest: Path, output: Path, epochs: int=20, batch_size: int=16, seq_len: int=8,
                 seed: int=17, device: str="cpu", patience: int=5, overwrite: bool=False) -> dict:
    if epochs < 1 or batch_size < 1 or seq_len < 1 or patience < 1: raise ValueError("epochs, batch_size, sequence length, and patience must be positive")
    if output.exists() and not overwrite: raise FileExistsError(f"refusing to overwrite checkpoint: {output}; pass --overwrite explicitly")
    metrics_path=output.with_suffix(".metrics.json")
    if metrics_path.exists() and not overwrite: raise FileExistsError(f"refusing to overwrite training report: {metrics_path}; pass --overwrite explicitly")
    started=time.perf_counter();manifest_hash=hashlib.sha256(manifest.read_bytes()).hexdigest()
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    with manifest.open(encoding="utf-8") as handle:
        for line_number,line in enumerate(handle,1):
            if not line.strip(): continue
            row=json.loads(line)
            if row.get("synthetic",False):
                raise ValueError("synthetic examples may be used for integration tests but are prohibited as real-data training evidence")
            ex=parse_example(row)
            if ex.schema_version != "ainak.aligned-example.v1" or ex.target_source != "human_reviewed":
                raise ValueError(f"line {line_number}: fusion training requires the reviewed v1 aligned-data workflow")
            for modality in MODALITIES:
                obs=getattr(ex,modality)
                if obs is not None and (not obs.model_checkpoint_id or not obs.preprocessing_version or not obs.feature_kind):
                    raise ValueError(f"line {line_number}: {modality} feature lacks checkpoint/preprocessing provenance")
    all_ds=FusionManifestDataset(manifest,seq_len)
    if not all_ds.items: raise ValueError("manifest has no usable aligned examples")
    ids=sorted(all_ds.split_groups)
    if len(ids)<3: raise ValueError("need at least three independent recording_id groups for train/validation/test separation")
    random.shuffle(ids); n_test=max(1,round(len(ids)*.2)); n_val=max(1,round(len(ids)*.2))
    if len(ids)-n_test-n_val<1: n_val=len(ids)-n_test-1
    test_ids=set(ids[:n_test]); val_ids=set(ids[n_test:n_test+n_val]); train_ids=set(ids[n_test+n_val:])
    train_ds=FusionManifestDataset(manifest,seq_len,train_ids); val_ds=FusionManifestDataset(manifest,seq_len,val_ids); test_ds=FusionManifestDataset(manifest,seq_len,test_ids)
    if not len(train_ds) or not len(val_ds) or not len(test_ds): raise ValueError("train, validation, or test split is empty")
    train_loader=DataLoader(train_ds,batch_size=batch_size,shuffle=True,collate_fn=collate)
    val_loader=DataLoader(val_ds,batch_size=batch_size,shuffle=False,collate_fn=collate)
    test_loader=DataLoader(test_ds,batch_size=batch_size,shuffle=False,collate_fn=collate)
    model=MultimodalTransformer(modality_dropout=.1).to(device); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    bce,ce=nn.BCELoss(),nn.CrossEntropyLoss(); best=float("inf"); best_state=None; stale_epochs=0
    history=[]
    for epoch in range(epochs):
        model.train(); train_loss=0.
        for features,mask,times,priority,confidence,alert in train_loader:
            features={k:v.to(device) for k,v in features.items()}; mask=mask.to(device); times=times.to(device); priority=priority.to(device); confidence=confidence.to(device); alert=alert.to(device)
            out=model(features,mask,times); loss=bce(out.priority,priority)+bce(out.confidence,confidence)+ce(out.alert_logits,alert)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step(); train_loss += float(loss.item())
        model.eval(); val_loss=0.; correct=0; count=0
        with torch.no_grad():
            for features,mask,times,priority,confidence,alert in val_loader:
                features={k:v.to(device) for k,v in features.items()}; mask=mask.to(device); times=times.to(device); priority=priority.to(device); confidence=confidence.to(device); alert=alert.to(device)
                out=model(features,mask,times); val_loss += float((bce(out.priority,priority)+bce(out.confidence,confidence)+ce(out.alert_logits,alert)).item())
                correct += int((out.alert_logits.argmax(-1)==alert).sum()); count += len(alert)
        val_loss/=max(len(val_loader),1); history.append({"epoch":epoch+1,"train_loss":train_loss/max(len(train_loader),1),"validation_loss":val_loss,"alert_accuracy":correct/max(count,1)})
        if val_loss < best:
            best=val_loss; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; stale_epochs=0
        else:
            stale_epochs+=1
            if stale_epochs >= patience: break
    model.load_state_dict(best_state); model.eval(); test_priority=[];test_confidence=[];test_logits=[];test_alert=[];test_confidence_targets=[]
    with torch.no_grad():
        for features,mask,times,priority,confidence,alert in test_loader:
            features={k:v.to(device) for k,v in features.items()};mask=mask.to(device);times=times.to(device)
            out=model(features,mask,times)
            test_priority.extend(out.priority.cpu().tolist());test_confidence.extend(out.confidence.cpu().tolist())
            test_logits.extend(out.alert_logits.cpu().tolist());test_alert.extend(alert.tolist());test_confidence_targets.extend(confidence.tolist())
    predicted=np.asarray(test_logits).argmax(axis=-1)
    heldout={"examples":len(test_alert),"priority_mae":float(np.mean(np.abs(np.asarray(test_priority)-np.asarray([x[-1][0][0].priority for x in test_ds.items])))),
        "confidence_mae":float(np.mean(np.abs(np.asarray(test_confidence)-np.asarray(test_confidence_targets)))),
        "alert_accuracy":float(np.mean(predicted==np.asarray(test_alert))),
        "alert_report":classification_report(test_alert,predicted,labels=list(range(len(ALERT_TYPES))),target_names=ALERT_TYPES,output_dict=True,zero_division=0),
        "confusion_matrix":confusion_matrix(test_alert,predicted,labels=list(range(len(ALERT_TYPES)))).tolist()}
    output.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"checkpoint_schema":"ainak.fusion-checkpoint.v1","state_dict":best_state,"model_config":{"dim":128,"heads":4,"layers":2,"ff_dim":256,"dropout":.1,"modality_dropout":.1},
        "feature_dims":DEFAULT_FEATURE_DIMS,"alert_types":ALERT_TYPES,"seed":seed,"sequence_length":seq_len,"training_examples":len(train_ds),"validation_examples":len(val_ds),"test_examples":len(test_ds),
        "recording_split":{"train":sorted(train_ids),"validation":sorted(val_ids),"test":sorted(test_ids)},"history":history,"trained_on_real_data":True,
        "training_data_provenance":"reviewed human-labeled ainak.aligned-example.v1 session manifest",
        "training_metadata":{"python":platform.python_version(),"torch":torch.__version__,"manifest_sha256":manifest_hash,"epochs_requested":epochs,"epochs_completed":len(history),"patience":patience,"batch_size":batch_size,"duration_seconds":time.perf_counter()-started}},output)
    alert_counts={name:0 for name in ALERT_TYPES}
    for _,window in all_ds.items: alert_counts[window[-1][0].alert_type]+=1
    report={"checkpoint":str(output),"training_examples":len(train_ds),"validation_examples":len(val_ds),"test_examples":len(test_ds),"best_validation_loss":best,
            "alert_type_distribution_over_windows":alert_counts,"recording_split":{"train":sorted(train_ids),"validation":sorted(val_ids),"test":sorted(test_ids)},
            "history":history,"heldout_test":heldout,"manifest_sha256":manifest_hash,"seed":seed,"duration_seconds":time.perf_counter()-started,
            "note":"Held-out metrics depend on independent reviewed annotations; the window count can exceed the number of independent recording groups."}
    metrics_path.write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report
