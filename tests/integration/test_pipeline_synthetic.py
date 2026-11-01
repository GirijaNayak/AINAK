"""Synthetic-only plumbing test; it is not model training or accuracy evidence."""
import numpy as np
from ainak.models.fusion.transformer import MultimodalTransformer
from ainak.runtime.events import ModelEvent
from ainak.runtime.pipeline import MultimodalPipeline


def test_synthetic_three_branch_fusion_and_missing_branch_fallback():
    model=MultimodalTransformer().eval()
    pipeline=MultimodalPipeline(model,trained=True,_test_only_allow_untrained=True)
    for name,size in (("vision",128),("audio",2048),("gesture",128)):
        pipeline.push(ModelEvent(name,10.,np.zeros(size,dtype=np.float32),.8,[]))
    result=pipeline.infer(10.1)
    assert result.transformer_used
    assert set(result.contributing_modalities)=={"vision","audio","gesture"}
    assert len(result.context)==128
    class SpeechSink:
        def __init__(self):self.messages=[]
        def say(self,text,priority=0):self.messages.append((text,priority));return True
    sink=SpeechSink();fallback=MultimodalPipeline(None,trained=False,speech=sink)
    fallback.push(ModelEvent("vision",12.,np.zeros(128,dtype=np.float32),.9,[{"class":"car","confidence":.9}]))
    out=fallback.infer(12.)
    assert not out.transformer_used and out.alert is not None and out.alert.fallback
    assert sink.messages==[(out.alert.text,int(round(out.alert.priority*100)))]
