"""AI AINAK Multimodal Assistive Intelligence Dashboard Package."""
from __future__ import annotations

__all__ = ["create_dashboard_app"]

def create_dashboard_app():
    from .app import app
    return app
