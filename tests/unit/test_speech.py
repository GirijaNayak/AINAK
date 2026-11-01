import threading

from ainak.audio_output.speech import LocalSpeech


def test_priority_queue_deduplicates_and_closes_worker():
    entered = threading.Event()
    release = threading.Event()
    spoken = []

    class FakeBackend:
        name = "fake-test-backend"
        def speak(self, text):
            if text == "first":
                entered.set()
                assert release.wait(3)
            spoken.append(text)
        def close(self):
            return None

    speaker = LocalSpeech(queue_capacity=4, duplicate_suppression_seconds=30,
                          _backend_factories=[FakeBackend])
    assert speaker.say("first", priority=0)
    assert entered.wait(2)
    assert speaker.say("low", priority=1)
    assert speaker.say("urgent", priority=10)
    assert not speaker.say(" URGENT ", priority=10)
    release.set()
    speaker.close(timeout=5)
    result = speaker.diagnostics()
    assert spoken == ["first", "urgent", "low"]
    assert result["spoken_count"] == 3
    assert result["deduplicated_count"] == 1
    assert result["worker_stopped"]


def test_full_queue_replaces_a_less_urgent_pending_alert():
    entered = threading.Event()
    release = threading.Event()
    spoken = []

    class FakeBackend:
        def speak(self, text):
            if text == "active":
                entered.set()
                assert release.wait(3)
            spoken.append(text)
        def close(self):
            return None

    speaker = LocalSpeech(queue_capacity=1, duplicate_suppression_seconds=0,
                          _backend_factories=[FakeBackend])
    assert speaker.say("active")
    assert entered.wait(2)
    assert speaker.say("low", priority=1)
    assert speaker.say("high", priority=9)
    release.set()
    speaker.close(timeout=5)
    assert spoken == ["active", "high"]
    assert speaker.diagnostics()["dropped_count"] == 1


def test_speech_backend_failure_is_reported_without_hanging():
    def fail():
        raise RuntimeError("backend unavailable")

    speaker = LocalSpeech(_backend_factories=[fail])
    speaker.close(timeout=2)
    result = speaker.diagnostics()
    assert not result["backend_ready"]
    assert "backend unavailable" in result["backend_error"]
    assert result["worker_stopped"]
