from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from ..data.schemas import ALERT_TYPES
from ..models.fusion.transformer import MultimodalTransformer,MODALITIES,DEFAULT_FEATURE_DIMS
from .fusion import FusionManifestDataset,collate
from ..runtime.checkpoint_validation import real_training_checkpoint_issues


class LateFusionBaseline(nn.Module):
    """Documented non-Transformer baseline: concatenate latest projected raw features plus availability bits."""
    def __init__(self):
        super().__init__(); width=sum(DEFAULT_FEATURE_DIMS.values())+3
        self.body=nn.Sequential(nn.Linear(width,256),nn.ReLU(),nn.Dropout(.1),nn.Linear(256,128),nn.ReLU())
        self.priority=nn.Sequential(nn.Linear(128,1),nn.Sigmoid());self.confidence=nn.Sequential(nn.Linear(128,1),nn.Sigmoid());self.alert=nn.Linear(128,len(ALERT_TYPES))
    def forward(self,features,mask):
        last=mask[:,-1,:]; joined=torch.cat([features[m][:,-1,:]*last[:,i:i+1] for i,m in enumerate(MODALITIES)]+[last.float()],dim=-1)
        h=self.body(joined)
        return self.priority(h).squeeze(-1),self.confidence(h).squeeze(-1),self.alert(h)


def _metrics(priorities,confidences,logits,targets,confidence_targets):
    pred=logits.argmax(-1).cpu().numpy();truth=targets.cpu().numpy()
    return {"priority_mae":float(torch.mean(torch.abs(priorities-targets)).item()),
        "confidence_mae":float(torch.mean(torch.abs(confidences-confidence_targets)).item()),
        "alert_accuracy":float(accuracy_score(truth,pred)),
        "alert_report":classification_report(truth,pred,labels=list(range(len(ALERT_TYPES))),target_names=ALERT_TYPES,output_dict=True,zero_division=0),
        "confusion_matrix":confusion_matrix(truth,pred,labels=list(range(len(ALERT_TYPES)))).tolist(),"samples":int(len(truth))}


def evaluate_fusion_ablations(manifest: Path,checkpoint: Path,output: Path|None=None,device: str="cpu",baseline_epochs: int=10)->dict:
    payload=torch.load(checkpoint,map_location=device,weights_only=False)
    issues=real_training_checkpoint_issues(payload,"fusion")
    if issues:raise ValueError("cannot evaluate an unverified fusion checkpoint: "+"; ".join(issues))
    split=payload.get("recording_split",{})
    if not split.get("test"):raise ValueError("checkpoint has no independent held-out test split; retrain with the reviewed Phase 3 workflow")
    train_ids=set(split["train"]);test_ids=set(split["test"])
    seq_len=int(payload.get("sequence_length",8));train=FusionManifestDataset(manifest,seq_len,train_ids);test=FusionManifestDataset(manifest,seq_len,test_ids)
    if not len(train) or not len(test):raise ValueError("stored train/test split no longer matches manifest")
    tr=DataLoader(train,batch_size=16,shuffle=True,collate_fn=collate);te=DataLoader(test,batch_size=32,collate_fn=collate)
    model=MultimodalTransformer(**payload.get("model_config",{})).to(device);model.load_state_dict(payload["state_dict"]);model.eval()
    subsets={"vision":[0],"audio":[1],"gesture":[2],"vision_audio":[0,1],"vision_gesture":[0,2],"audio_gesture":[1,2],"all_three":[0,1,2]}
    results={}
    for name,indices in subsets.items():
        p=[];c=[];l=[];yt=[];ct=[]
        with torch.no_grad():
            for f,m,t,priority,confidence,alert in te:
                f={k:v.to(device) for k,v in f.items()};m=m.to(device);t=t.to(device)
                reduced=m.clone();reduced[:,:, [i for i in range(3) if i not in indices]]=False
                out=model(f,reduced,t);p.append(out.priority.cpu());c.append(out.confidence.cpu());l.append(out.alert_logits.cpu());yt.append(alert);ct.append(confidence)
        results[name]=_metrics(torch.cat(p),torch.cat(c),torch.cat(l),torch.cat(yt),torch.cat(ct))
    baseline=LateFusionBaseline().to(device);opt=torch.optim.AdamW(baseline.parameters(),lr=1e-3);bce=nn.BCELoss();ce=nn.CrossEntropyLoss()
    for _ in range(baseline_epochs):
        baseline.train()
        for f,m,t,p,c,y in tr:
            f={k:v.to(device) for k,v in f.items()};m=m.to(device);p=p.to(device);c=c.to(device);y=y.to(device)
            bp,bc,bl=baseline(f,m);loss=bce(bp,p)+bce(bc,c)+ce(bl,y);opt.zero_grad();loss.backward();opt.step()
    baseline.eval();p=[];c=[];l=[];yt=[];ct=[]
    with torch.no_grad():
        for f,m,t,priority,confidence,alert in te:
            f={k:v.to(device) for k,v in f.items()};m=m.to(device);bp,bc,bl=baseline(f,m);p.append(bp.cpu());c.append(bc.cpu());l.append(bl.cpu());yt.append(alert);ct.append(confidence)
    results["late_fusion_mlp_baseline"]=_metrics(torch.cat(p),torch.cat(c),torch.cat(l),torch.cat(yt),torch.cat(ct))
    report={"checkpoint":str(checkpoint),"split_groups":payload["recording_split"],"heldout_test_ablations":results,
        "synthetic_data_used":False,"note":"All modality subsets and the MLP baseline use the same untouched held-out test split; interpret only with independent human-reviewed recordings."}
    if output:output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report
