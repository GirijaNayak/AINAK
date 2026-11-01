import numpy as np
import pytest
from types import SimpleNamespace
import torch
from ainak.models.fusion.transformer import MultimodalTransformer
from ainak.models.gesture.lstm import GestureLSTM,GestureSequenceBuffer,normalize_landmarks,FEATURES
from ainak.models.audio.classifier import prepare_audio_window


def test_fusion_projects_all_modalities_and_masks_missing():
    model=MultimodalTransformer().eval()
    features={"vision":torch.randn(2,3,128),"audio":torch.randn(2,3,2048),"gesture":torch.randn(2,3,128)}
    mask=torch.tensor([[[1,1,1],[1,0,0],[0,1,1]],[[0,0,0],[0,0,0],[0,0,0]]],dtype=torch.bool)
    out=model(features,mask,torch.zeros(2,3))
    assert out.context.shape==(2,128)
    assert out.alert_logits.shape==(2,4)
    assert torch.allclose(out.context[1],torch.zeros(128))
    assert torch.isfinite(out.context).all()


def test_fusion_invalid_shapes_rejected():
    model=MultimodalTransformer()
    with pytest.raises(ValueError):model({},torch.ones(1,3),None)
    with pytest.raises(ValueError):model({"vision":torch.randn(1,2,127)},torch.ones(1,2,3),None)


def test_gesture_lstm_and_normalization():
    model=GestureLSTM(); logits,embedding=model(torch.randn(4,24,FEATURES))
    assert logits.shape==(4,4) and embedding.shape==(4,128)
    points=np.random.randn(33,4).astype(np.float32); points[:,3]=1.
    normalized=normalize_landmarks(points)
    assert normalized.shape==(FEATURES,) and np.isfinite(normalized).all()


def test_gesture_sequence_buffer_waits_for_full_window():
    buffer=GestureSequenceBuffer(3)
    assert buffer.append(np.zeros(FEATURES)) is None
    assert buffer.append(np.zeros(FEATURES)) is None
    assert buffer.append(np.zeros(FEATURES)).shape==(3,FEATURES)


def test_pose_tracker_reports_missing_tasks_model_for_current_mediapipe(monkeypatch):
    import sys
    from types import SimpleNamespace
    from ainak.models.gesture.pose import create_pose_tracker
    monkeypatch.setitem(sys.modules,"mediapipe",SimpleNamespace(__version__="0.10.31"))
    with pytest.raises(RuntimeError,match=r"provide a local.*\.task model"):
        create_pose_tracker()


def test_pose_diagnostic_reports_tasks_api_and_missing_model(monkeypatch, tmp_path):
    import sys
    from ainak.models.gesture.pose import pose_readiness
    monkeypatch.setitem(sys.modules, "mediapipe", SimpleNamespace(
        __version__="0.10.31", tasks=SimpleNamespace(vision=object())))
    result = pose_readiness(tmp_path / "missing.task")
    assert result["selected_api"] == "tasks_pose_landmarker"
    assert not result["model_bundle"]["available"]
    assert not result["extraction_ready"]
    assert not result["pose_extraction_verified"]


def test_pose_model_acquirer_validates_bundle_and_records_hash(monkeypatch, tmp_path):
    import io
    import zipfile
    import ainak.models.gesture.pose as pose_module

    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as bundle:
        bundle.writestr("pose_landmarker.tflite", b"fixture model bytes")

    class FakeResponse:
        def __init__(self, content):
            self.content = io.BytesIO(content)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return None
        def read(self, size=-1):
            return self.content.read(size)

    monkeypatch.setattr(pose_module.urllib.request, "urlopen", lambda *args, **kwargs: FakeResponse(payload.getvalue()))
    output = tmp_path / "pose_landmarker_lite.task"
    report = pose_module.acquire_pose_model(output)
    assert report["bundle_valid"]
    assert report["bytes"] == output.stat().st_size
    assert report["sha256"].startswith("sha256:")
    assert output.with_suffix(".task.source.json").is_file()


def test_audio_input_validation_and_silence():
    clipped=prepare_audio_window(np.full(8000,2.,dtype=np.float32),32000)
    assert clipped.max()==1.
    with pytest.raises(ValueError,match="silence"):prepare_audio_window(np.zeros(8000),32000)
    with pytest.raises(ValueError,match="32 kHz"):prepare_audio_window(np.ones(8000),16000)


def test_late_fusion_baseline_shape():
    from ainak.training.evaluation import LateFusionBaseline
    model=LateFusionBaseline()
    features={"vision":torch.randn(2,4,128),"audio":torch.randn(2,4,2048),"gesture":torch.randn(2,4,128)}
    p,c,alerts=model(features,torch.ones(2,4,3,dtype=torch.bool))
    assert p.shape==(2,) and c.shape==(2,) and alerts.shape==(2,4)


def test_vision_missing_weights_fails_explicitly(tmp_path):
    from ainak.models.vision.detector import VisionDetector
    with pytest.raises(FileNotFoundError,match="acquire vision"):
        VisionDetector(tmp_path/"missing.pt")


def test_audio_missing_weights_fails_explicitly(tmp_path):
    from ainak.models.audio.classifier import PannsCNN14
    with pytest.raises(FileNotFoundError,match="acquire audio"):
        PannsCNN14(tmp_path/"missing.pth")
