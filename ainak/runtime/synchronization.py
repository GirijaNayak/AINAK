from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import math
from typing import Iterable
from .events import ModelEvent


@dataclass
class SynchronizedFrame:
    timestamp: float
    events: dict[str, ModelEvent | None]


class TemporalSynchronizer:
    """Retains the newest observation per modality and masks stale/out-of-window inputs."""
    def __init__(self, stale_after_seconds: float = 3.0, window_seconds: float = 2.0):
        if stale_after_seconds <= 0 or window_seconds <= 0: raise ValueError("time limits must be positive")
        self.stale_after_seconds, self.window_seconds = stale_after_seconds, window_seconds
        self._events: dict[str, deque[ModelEvent]] = {k: deque(maxlen=128) for k in ("vision", "audio", "gesture")}

    def push(self, event: ModelEvent) -> None:
        if event.modality not in self._events: raise ValueError(f"unknown modality {event.modality}")
        if not isinstance(event.timestamp, (int, float)) or not math.isfinite(event.timestamp) or event.timestamp < 0:
            raise ValueError("event timestamp must be finite nonnegative session seconds")
        if not isinstance(event.confidence, (int, float)) or not math.isfinite(event.confidence) or not 0 <= event.confidence <= 1:
            raise ValueError("event confidence must be finite and in [0,1]")
        if not isinstance(event.latency_ms, (int, float)) or not math.isfinite(event.latency_ms) or event.latency_ms < 0:
            raise ValueError("event latency_ms must be finite and nonnegative")
        if self._events[event.modality] and event.timestamp < self._events[event.modality][-1].timestamp:
            raise ValueError("events must arrive in nondecreasing timestamp order per modality")
        self._events[event.modality].append(event)

    def align(self, timestamp: float) -> SynchronizedFrame:
        if not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("alignment timestamp must be finite nonnegative session seconds")
        aligned: dict[str, ModelEvent | None] = {}
        for name, queue in self._events.items():
            candidate = next((e for e in reversed(queue) if e.timestamp <= timestamp), None)
            if candidate is None or timestamp - candidate.timestamp > self.stale_after_seconds or timestamp - candidate.timestamp > self.window_seconds:
                aligned[name] = None
            else:
                aligned[name] = candidate
        return SynchronizedFrame(timestamp, aligned)

    def history(self, timestamp: float, max_steps: int = 8, lookback_seconds: float = 8.) -> list[SynchronizedFrame]:
        """Return timestamp anchors with modality observations aligned independently."""
        if max_steps < 1 or lookback_seconds <= 0:
            raise ValueError("max_steps and lookback_seconds must be positive")
        cutoff = timestamp - lookback_seconds
        anchors = sorted({event.timestamp for queue in self._events.values() for event in queue
                          if cutoff <= event.timestamp <= timestamp})
        anchors.append(timestamp)
        anchors = sorted(set(anchors))[-max_steps:]
        return [self.align(anchor) for anchor in anchors]

    def clear(self) -> None:
        for queue in self._events.values(): queue.clear()
