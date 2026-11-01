import pytest
from ainak.config import RuntimeConfig
from ainak.runtime.events import ModelEvent
from ainak.runtime.synchronization import TemporalSynchronizer
from ainak.decision.alert_engine import AlertEngine


def test_sync_drops_stale_events_and_keeps_missing_none():
    sync=TemporalSynchronizer(stale_after_seconds=2,window_seconds=1)
    sync.push(ModelEvent("vision",10,[0.]))
    assert sync.align(10.5).events["vision"] is not None
    assert sync.align(12).events["vision"] is None
    assert sync.align(10.5).events["audio"] is None


def test_sync_rejects_out_of_order_event():
    sync=TemporalSynchronizer();sync.push(ModelEvent("audio",2,[0.]))
    with pytest.raises(ValueError):sync.push(ModelEvent("audio",1,[0.]))


def test_alert_stale_low_confidence_and_cooldown():
    engine=AlertEngine(min_confidence=.5,cooldown_seconds=3)
    assert engine.decide([{"class":"car","confidence":.9,"timestamp":0,"modality":"vision"}],10) is None
    alert=engine.decide([{"class":"car","confidence":.9,"timestamp":10,"modality":"vision"}],10)
    assert alert and alert.fallback and alert.sources==("vision",)
    assert engine.decide([{"class":"car","confidence":.9,"timestamp":11,"modality":"vision"}],11) is None


def test_configuration_validation():
    assert RuntimeConfig().validate()
    with pytest.raises(ValueError):RuntimeConfig(vision_confidence=2).validate()
    with pytest.raises(ValueError):RuntimeConfig(device="bogus").validate()
