"""CLI entrypoint for running the AI AINAK dashboard server."""
from __future__ import annotations

import argparse
import sys
import threading
import time
import webbrowser
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="AI AINAK – Multimodal Assistive Intelligence Console Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--no-open", action="store_true", help="Do not automatically open default web browser")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"
    print("=" * 72)
    print("   AI AINAK – MULTIMODAL ASSISTIVE INTELLIGENCE CONSOLE")
    print(f"   Starting server at: {url}")
    print("   Press Ctrl+C to terminate.")
    print("=" * 72)

    if not args.no_open:
        def open_browser():
            time.sleep(1.2)
            try:
                webbrowser.open(url)
            except Exception:
                pass
        threading.Thread(target=open_browser, daemon=True).start()

    uvicorn.run("ainak.interfaces.dashboard.app:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
