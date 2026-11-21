"""Check readable product-label copy against the reference in every frame of a product shot."""

import argparse
import gzip
import json
from pathlib import Path

import cv2

from .audit import norm, read_text
from .campaign_edit import fingerprint
from .core import save


def verify(reference, video, phrases, report, start=0.0, seconds=None, minimum=0.90):
    wanted = {phrase: norm(phrase) for phrase in phrases}
    if not wanted or not all(wanted.values()) or not 0 < minimum <= 1:
        raise ValueError("Provide product words and a required readable-frame fraction")

    def matches(image):
        seen = [
            norm(item["text"])
            for item in read_text(image)
            if item["confidence"] >= 0.65
        ]
        return {
            key: any(value in line for line in seen) for key, value in wanted.items()
        }

    im = cv2.imread(str(reference))
    if im is None or not all(matches(im).values()):
        raise ValueError(
            "The expected package words must be readable in the source reference"
        )
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not capture.isOpened() or fps <= 0:
        capture.release()
        raise ValueError(f"Unreadable package shot: {video}")
    first = round(start * fps)
    available = round(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    count = available - first if seconds is None else round(seconds * fps)
    if first < 0 or count <= 0 or first + count > available:
        capture.release()
        raise ValueError("Label interval is outside the source")
    capture.set(cv2.CAP_PROP_POS_FRAMES, first)
    counts = dict.fromkeys(wanted, 0)
    ledger = report.with_suffix(".frames.jsonl.gz")
    try:
        with gzip.open(ledger, "wt", encoding="utf8") as log:
            for index in range(first, first + count):
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f"Could not decode frame {index}")
                found = matches(frame)
                for key, hit in found.items():
                    counts[key] += int(hit)
                log.write(
                    json.dumps({"frame": index, "readable": found}, ensure_ascii=False)
                    + "\n"
                )
    finally:
        capture.release()
    result = {
        "passed": all(hits / count >= minimum for hits in counts.values()),
        "frames": count,
        "matches": counts,
        "minimum_fraction": minimum,
        "source_sha256": fingerprint(reference),
        "video_sha256": fingerprint(video),
        "start_frame": first,
        "ledger": ledger.name,
    }
    save(report, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("reference", "video", "report"):
        parser.add_argument(arg, type=Path)
    parser.add_argument("phrases", nargs="+")
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--minimum", type=float, default=0.9)
    args = parser.parse_args()
    result = verify(
        args.reference,
        args.video,
        args.phrases,
        args.report,
        args.start,
        args.seconds,
        args.minimum,
    )
    print(result, flush=True)
    raise SystemExit(0 if result["passed"] else 2)
