import argparse
import json
import threading
import time
import webbrowser
from pathlib import Path

import httpx
import uvicorn

from . import config
from .audit import audit
from .core import Brief, create_job, jobdir, read, record, save
from .director import critique
from .pipeline import make
from .providers import inventory


def main():
    p = argparse.ArgumentParser(prog="framecraft")
    commands = p.add_subparsers(dest="action", required=True)
    server = commands.add_parser("studio")
    server.add_argument(
        "--open", action="store_true", help="Open the ready studio in your browser"
    )
    server.add_argument("--port", type=int, default=7865)
    commands.add_parser("models")
    check = commands.add_parser("analyze")
    check.add_argument("video", type=Path)
    check.add_argument("--out", type=Path, required=True)
    check.add_argument(
        "--captions", type=Path, help="JSON cue list with start, end and text"
    )
    produce = commands.add_parser("create")
    produce.add_argument("brief")
    produce.add_argument("--variants", type=int, default=2)
    produce.add_argument("--shots", type=int, default=3)
    produce.add_argument("--language", choices=["tr", "en"], default="tr")
    args = p.parse_args()
    config.setup()
    if args.action == "studio":
        if args.open:
            url = f"http://127.0.0.1:{args.port}"

            def ready():
                try:
                    return (
                        httpx.get(url + "/openapi.json", timeout=1).json()["info"][
                            "title"
                        ]
                        == "UGC Creator"
                    )
                except (ValueError, KeyError, httpx.HTTPError):
                    return False

            if ready():
                webbrowser.open(url)
                return

            def open_when_ready():
                for _ in range(45):
                    if ready():
                        webbrowser.open(url)
                        return
                    time.sleep(1)

            threading.Thread(target=open_when_ready, daemon=True).start()
        uvicorn.run("framecraft.server:app", host="127.0.0.1", port=args.port)
    elif args.action == "models":
        print(json.dumps(inventory(), ensure_ascii=False, indent=2))
    elif args.action == "analyze":
        cues = read(args.captions) if args.captions else []
        report = audit(
            args.video, args.out, "master" if cues else "reference", cues, print
        )
        save(args.out / "critique.json", critique(args.out, report))
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        brief = Brief(
            brief=args.brief,
            variants=args.variants,
            shots=args.shots,
            language=args.language,
        )
        job = create_job("create")
        save(jobdir(job) / "brief.json", brief.model_dump())
        record(job, "running", "Starting CLI job")
        try:
            make(job)
        except Exception as exc:
            record(job, "failed", str(exc))
            raise
        print(jobdir(job))
