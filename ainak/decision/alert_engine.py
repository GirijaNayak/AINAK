from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass
class Alert:
    text: str
    priority: float
    timestamp: float
    sources: tuple[str, ...]
    kind: str
    fallback: bool = False


class AlertEngine:
    """Deterministic event policy with staleness, persistence, dedupe, and cooldown."""
    PRIORITY = {"person": .45, "car": .75, "truck": .8, "bus": .75, "bicycle": .65, "motorcycle": .7,
                "siren": .95, "alarm": .9, "vehicle": .8, "stop": .7, "pointing": .4, "waving": .2}
    def __init__(self, min_confidence: float = .55, cooldown_seconds: float = 5., stale_seconds: float = 3., persistence: int = 1):
        self.min_confidence, self.cooldown_seconds, self.stale_seconds, self.persistence = min_confidence, cooldown_seconds, stale_seconds, persistence
        self._last: dict[str,float] = {}; self._counts: dict[str,int] = {}

    def decide(self, observations: list[dict[str, Any]], now: float, fusion: dict[str, Any] | None = None) -> Alert | None:
        candidates = []
        for item in observations:
            ts = float(item.get("timestamp", now)); conf = float(item.get("confidence",0)); label = str(item.get("class", item.get("label","unknown"))).lower()
            if now-ts > self.stale_seconds or ts > now or conf < self.min_confidence: continue
            priority = self.PRIORITY.get(label, .25) * conf
            if priority >= .25: candidates.append((priority,label,item.get("modality","unknown")))
        if fusion and fusion.get("trained",False):
            candidates.append((float(fusion.get("priority",0))*float(fusion.get("confidence",0)), str(fusion.get("alert_type","context")), "fusion"))
        if not candidates: return None
        priority,label,source=max(candidates)
        key=f"{source}:{label}"
        self._counts[key] = self._counts.get(key,0)+1
        if self._counts[key] < self.persistence: return None
        if now-self._last.get(key, -1e9) < self.cooldown_seconds: return None
        self._last[key]=now
        description = {"car":"vehicle", "truck":"large vehicle", "bus":"bus", "motorcycle":"motorcycle", "bicycle":"bicycle", "person":"person"}.get(label,label)
        text = f"{description} detected" if source == "vision" else f"{description} sound detected" if source == "audio" else f"{description} gesture detected"
        return Alert(text, priority, now, (source,), "urgent" if priority >= .7 else "caution", fallback=source != "fusion")
