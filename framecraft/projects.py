"""Keep numbered, immutable UGC edits together in the local Framecraft library."""

import hashlib
from html import escape
import os
import re
import shutil
import tempfile
from pathlib import Path
from urllib.parse import quote
from filelock import FileLock
from .core import home, read, save


def projects_root():
    return Path(os.environ.get("FRAMECRAFT_PROJECTS", home() / "projects")).resolve()


def project_folder(name):
    label = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', " ", name)
    label = " ".join(label.split())[:72].strip(" .")
    if not label or Path(label).is_reserved():
        raise ValueError("Choose a project name that can be used as a folder")
    root = projects_root()
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve(strict=True)
    folder = (root / label).resolve()
    if folder.parent != root:
        raise ValueError("Invalid project folder")
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def editions(folder):
    records = []
    for entry in folder.glob("* - UGC */iteration.json"):
        record = read(entry)
        if (
            entry.parent.name == f"{folder.name} - UGC {record['number']:03d}"
            and (entry.parent / (entry.parent.name + ".mp4")).is_file()
        ):
            records.append(record)
    return sorted(records, key=lambda record: record["number"], reverse=True)


def archive(project, video, *, stage="master", report=None, ledger=None):
    folder = project_folder(project)
    with video.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if report and read(report).get("sha256", digest) != digest:
        raise ValueError("Assessment belongs to a different MP4")
    with FileLock(str(folder / ".iterations.lock")):
        records = editions(folder)
        for record in records:
            if record["sha256"] == digest:
                return folder / record["title"] / (record["title"] + ".mp4")
        number = max((e["number"] for e in records), default=0) + 1
        title = f"{folder.name} - UGC {number:03d}"
        with tempfile.TemporaryDirectory(dir=folder) as tmp:
            stage_dir = Path(tmp) / title
            stage_dir.mkdir()
            shutil.copyfile(video, stage_dir / (title + ".mp4"))
            for src, name in ((report, "assessment.json"), (ledger, "frames.jsonl.gz")):
                if src:
                    shutil.copyfile(src, stage_dir / name)
            save(
                stage_dir / "iteration.json",
                {"title": title, "number": number, "stage": stage, "sha256": digest},
            )
            os.rename(stage_dir, folder / title)
    write_gallery()
    return folder / title / (title + ".mp4")


def library():
    root = projects_root()
    if not root.exists():
        return []
    return [
        {"name": p.name, "editions": editions(p)}
        for p in sorted(root.iterdir())
        if p.is_dir() and p.resolve().parent == root.resolve()
    ]


def write_gallery():
    root = projects_root()
    with FileLock(str(root / ".gallery.lock")):
        sections = []
        for project in library():
            videos = []
            for edition in project["editions"]:
                title = edition["title"]
                url = "/".join(
                    quote(part, safe="")
                    for part in (project["name"], title, title + ".mp4")
                )
                stage = {"preview": " · Ön izleme", "archive": " · Arşiv"}.get(
                    edition["stage"], ""
                )
                videos.append(
                    f'<details><summary>UGC {edition["number"]:03d}{stage}</summary><video controls playsinline preload="none" src="{url}"></video></details>'
                )
            if videos:
                sections.append(
                    f"<section><h2>{escape(project['name'])}</h2>{''.join(videos)}</section>"
                )
        page = (
            '<!doctype html><html lang="tr"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>UGC Creator · Projeler</title><style>body{max-width:68rem;margin:2rem auto;padding:0 1rem;background:#131618;color:#eee;font:1rem system-ui}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(18rem,100%),1fr));gap:1rem;align-items:start}section{padding:1.3rem;background:#202429;border-radius:.6rem}h2{margin-top:0;font-size:1.2rem}summary{padding:.55rem 0;cursor:pointer}video{width:100%;max-height:65vh;background:#0c0e10}</style><h1>UGC Creator · Projeler</h1><main>'
            + "".join(sections)
            + "</main></html>"
        )
        temp = root / ".index.tmp"
        temp.write_text(page, encoding="utf-8")
        os.replace(temp, root / "index.html")
