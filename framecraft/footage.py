import json

from .core import read
from .edit import ffmpeg


def cut(source, analysis, index, total, variant, attempt, target):
    report = read(analysis / "report.json")
    duration = report["duration"]
    if duration * report["fps"] < total * 3:
        raise ValueError("Reference is too short for this shot count")
    window = duration / total
    left = index * window
    length = min(3.04, window)
    right = left + window - length
    candidates = []
    with (analysis / "frames.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if left <= row["time"] <= right and not row["flags"]:
                candidates.append((min(row["motion"], 28), row["time"]))
    candidates.sort(reverse=True)
    # Choose from the strongest moving frames while retaining chronological beat order.
    top = candidates[: max(1, len(candidates) // 3)]
    start = top[(variant * 13 + attempt * 17) % len(top)][1] if top else left
    ffmpeg(
        [
            "-ss",
            f"{start:.6f}",
            "-i",
            source,
            "-t",
            f"{length:.6f}",
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-vf",
            "fps=24",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "veryfast",
            "-c:a",
            "aac",
            target,
        ]
    )
    return target, round(start, 4)
