from __future__ import annotations

import heapq
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from typing import Any


class _Pyttsx3Backend:
    name = "pyttsx3"

    def __init__(self, rate: int, voice: str | None):
        import pyttsx3
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)
        if voice:
            self.engine.setProperty("voice", voice)

    def speak(self, text: str) -> None:
        self.engine.say(text)
        self.engine.runAndWait()

    def close(self) -> None:
        self.engine.stop()


class _SystemSpeechBackend:
    """Windows .NET System.Speech fallback for machines where pyttsx3/SAPI5 fails."""
    name = "powershell-system.speech"

    def __init__(self, voice: str | None):
        self.executable = shutil.which("powershell.exe") or shutil.which("powershell")
        if not self.executable:
            raise RuntimeError("Windows PowerShell is unavailable")
        probe = (
            "Add-Type -AssemblyName System.Speech; "
            "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "$v=@($s.GetInstalledVoices() | Where-Object {$_.Enabled} | ForEach-Object {$_.VoiceInfo.Name}); "
            "$s.Dispose(); if($v.Count -lt 1){exit 4}; 'AINAK_READY'"
        )
        result = subprocess.run([self.executable, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", probe],
            capture_output=True, text=True, timeout=10, check=False)
        if result.returncode != 0 or "AINAK_READY" not in result.stdout:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(f"System.Speech initialization failed: {detail or result.returncode}")
        self.voice = voice

    def speak(self, text: str) -> None:
        # Pass alert content through environment variables; never interpolate it as PowerShell code.
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "if($env:AINAK_SPEECH_VOICE){$s.SelectVoice($env:AINAK_SPEECH_VOICE)}; "
            "$s.Speak($env:AINAK_SPEECH_TEXT); $s.Dispose(); exit 0"
        )
        env = os.environ.copy()
        env["AINAK_SPEECH_TEXT"] = text
        env["AINAK_SPEECH_VOICE"] = self.voice or ""
        result = subprocess.run([self.executable, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", script],
            env=env, capture_output=True, text=True, timeout=30, check=False)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(f"System.Speech playback failed: {detail or result.returncode}")

    def close(self) -> None:
        return None


class LocalSpeech:
    """Nonblocking, prioritized local speech queue with duplicate suppression."""

    def __init__(self, rate: int = 185, voice: str | None = None, *, queue_capacity: int = 8,
                 duplicate_suppression_seconds: float = 2.0,
                 _backend_factories: list[Callable[[], Any]] | None = None):
        if queue_capacity < 1 or duplicate_suppression_seconds < 0:
            raise ValueError("queue_capacity must be positive and duplicate suppression nonnegative")
        self.rate, self.voice = rate, voice
        self.queue_capacity = queue_capacity
        self.duplicate_suppression_seconds = duplicate_suppression_seconds
        self._backend_factories = _backend_factories or [
            lambda: _Pyttsx3Backend(rate, voice),
            lambda: _SystemSpeechBackend(voice),
        ]
        self._condition = threading.Condition()
        self._queue: list[tuple[int, int, str]] = []
        self._sequence = 0
        self._recent_text: dict[str, float] = {}
        self._closing = False
        self.backend_name: str | None = None
        self.backend_error: str | None = None
        self.backend_warnings: list[str] = []
        self.backend_ready = False
        self.spoken_count = 0
        self.deduplicated_count = 0
        self.dropped_count = 0
        self._worker = threading.Thread(target=self._run, name="ainak-speech", daemon=True)
        self._worker.start()

    def say(self, text: str, priority: int = 0) -> bool:
        if not isinstance(text, str) or not text.strip():
            return False
        if isinstance(priority, bool) or not isinstance(priority, int):
            raise ValueError("speech priority must be an integer; higher values are more urgent")
        normalized = " ".join(text.casefold().split())
        now = time.monotonic()
        with self._condition:
            if self._closing or self.backend_error is not None:
                return False
            last = self._recent_text.get(normalized)
            if last is not None and now - last < self.duplicate_suppression_seconds:
                self.deduplicated_count += 1
                return False
            self._recent_text[normalized] = now
            self._recent_text = {key: stamp for key, stamp in self._recent_text.items()
                                 if now - stamp < max(self.duplicate_suppression_seconds * 4, 30.)}
            self._sequence += 1
            item = (-priority, self._sequence, text.strip())
            if len(self._queue) >= self.queue_capacity:
                lowest_index = max(range(len(self._queue)), key=lambda index: self._queue[index][0])
                queued_lowest_priority = -self._queue[lowest_index][0]
                if priority <= queued_lowest_priority:
                    self.dropped_count += 1
                    return False
                self._queue.pop(lowest_index)
                heapq.heapify(self._queue)
                self.dropped_count += 1
            heapq.heappush(self._queue, item)
            self._condition.notify()
            return True

    def _run(self) -> None:
        backend = None
        for factory in self._backend_factories:
            try:
                backend = factory()
                self.backend_name = getattr(backend, "name", type(backend).__name__)
                self.backend_ready = True
                break
            except Exception as exc:
                self.backend_warnings.append(f"{getattr(factory, '__name__', 'backend')}: {type(exc).__name__}: {exc}")
        if backend is None:
            self.backend_error = "; ".join(self.backend_warnings) or "no local speech backend is configured"
            with self._condition:
                self.dropped_count += len(self._queue)
                self._queue.clear()
                self._condition.notify_all()
            return
        try:
            while True:
                with self._condition:
                    while not self._queue and not self._closing:
                        self._condition.wait()
                    if not self._queue and self._closing:
                        break
                    _, _, text = heapq.heappop(self._queue)
                try:
                    backend.speak(text)
                    self.spoken_count += 1
                except Exception as exc:
                    self.backend_error = f"{type(exc).__name__}: {exc}"
                    with self._condition:
                        self.dropped_count += len(self._queue)
                        self._queue.clear()
                        self._closing = True
                        self._condition.notify_all()
                    break
        finally:
            try:
                backend.close()
            except Exception as exc:
                self.backend_warnings.append(f"backend cleanup: {type(exc).__name__}: {exc}")

    def close(self, timeout: float = 35.) -> None:
        with self._condition:
            self._closing = True
            self._condition.notify_all()
        self._worker.join(timeout=max(0., timeout))
        if self._worker.is_alive():
            self.backend_error = self.backend_error or "speech worker did not finish before shutdown timeout"

    def diagnostics(self) -> dict:
        return {"backend": self.backend_name, "backend_ready": self.backend_ready,
            "spoken_count": self.spoken_count, "deduplicated_count": self.deduplicated_count,
            "dropped_count": self.dropped_count, "backend_error": self.backend_error,
            "backend_warnings": list(self.backend_warnings), "worker_stopped": not self._worker.is_alive()}
