"""Find a continuously moving, visibly relevant detail within an existing generated take."""

import argparse
import os
from pathlib import Path
import subprocess
import tempfile

import cv2
import numpy as np
from pydantic import BaseModel

from .campaign_edit import fingerprint
from .core import save
from .director import ask
from .motion_campaign import moving_interval
from .motion_review import reference_check


class Ranking(BaseModel):
    windows: list[int]


class Subject(BaseModel):
    visible: bool
    observation: str


def windows(width, height):
    if min(width, height) < 16:
        raise ValueError("Source is too small for a detail crop")
    result = []
    for fraction in (0.45, 0.62, 0.8):
        w, h = int(width * fraction) // 2 * 2, int(height * fraction) // 2 * 2
        for x in (0, 0.5, 1):
            for y in (0, 0.5, 1):
                box = (
                    round((width - w) * x) // 2 * 2,
                    round((height - h) * y) // 2 * 2,
                    w,
                    h,
                )
                if box not in result:
                    result.append(box)
    return result


def samples(video):
    cap = cv2.VideoCapture(str(video))
    results = []
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for fraction in (0.18, 0.5, 0.82):
            cap.set(cv2.CAP_PROP_POS_FRAMES, round((count - 1) * fraction))
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Could not decode {video}")
            results.append(frame)
    finally:
        cap.release()
    return results


def select(source, target, seconds, direction, model=None):
    if source.resolve() == target.resolve() or seconds <= 0 or not direction.strip():
        raise ValueError(
            "Choose a distinct output, a visible subject, and a positive duration"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    original = fingerprint(source)
    frames = samples(source)
    height, width = frames[0].shape[:2]
    options = windows(width, height)
    crops = []
    for i, (x, y, w, h) in enumerate(options):
        tile = np.zeros((156, 144, 3), np.uint8)
        for n, frame in enumerate((frames[0], frames[2])):
            tile[24:, n * 72 : (n + 1) * 72] = cv2.resize(
                frame[y : y + h, x : x + w], (72, 132)
            )
        cv2.putText(
            tile,
            str(i + 1),
            (3, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )
        crops.append(tile)
    while len(crops) % 3:
        crops.append(np.zeros_like(crops[0]))
    board = np.vstack([np.hstack(crops[i : i + 3]) for i in range(0, len(crops), 3)])
    order = ask(
        Ranking,
        "Each numbered candidate shows two times through a different physical framing of a real moving source. Rank up to seven distinct windows that best show the requested action, have sustained visible movement and avoid faces, visibly malformed limbs, and static background. Use the visible contents and return numbered choices.",
        {"action": direction},
        cv2.imencode(".jpg", board)[1].tobytes(),
        model_override=model,
    )
    attempts = []
    with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
        for n in dict.fromkeys(order.windows):
            if n < 1 or n > len(options):
                continue
            x, y, w, h = options[n - 1]
            cut = Path(tmp) / f"view-{n}.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-i",
                    str(source),
                    "-vf",
                    f"crop={w}:{h}:{x}:{y},scale={width}:{height}:flags=lanczos",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "15",
                    "-threads",
                    "3",
                    str(cut),
                ],
                check=True,
            )
            at = moving_interval(cut, seconds, skip=0.1)
            item = {"box": [x, y, w, h], "moving_start": at}
            attempts.append(item)
            if at is None:
                continue
            check = reference_check(cut, None, model=model)
            item["visual"] = check
            if check["defect"] != "none":
                continue
            actual = np.hstack([cv2.resize(f, (200, 340)) for f in samples(cut)])
            subject = ask(
                Subject,
                "Inspect the three successive views and verify the visible subject and action against the request. A useful wardrobe or product detail must fill the view, hold coherent anatomy when a hand is visible, and show a meaningful change between the samples. Mark false for an empty room or an unrelated garment or object. Describe the specific evidence.",
                {"action": direction},
                cv2.imencode(".jpg", actual)[1].tobytes(),
                model_override=model,
            )
            item["subject"] = subject.model_dump()
            if not subject.visible:
                continue
            if original != fingerprint(source):
                raise ValueError("The source changed during selection")
            os.replace(cut, target)
            save(target.with_suffix(".review.json"), check)
            save(
                target.with_suffix(".reframe.json"),
                {
                    "source": str(source),
                    "source_sha256": original,
                    "output_sha256": fingerprint(target),
                    "seconds": seconds,
                    "direction": direction,
                    "selection": item,
                    "attempts": attempts,
                },
            )
            return target
    save(
        target.with_suffix(".attempts.json"),
        {"source_sha256": original, "attempts": attempts},
    )
    raise ValueError("No validated moving view in this source")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("seconds", type=float)
    parser.add_argument("direction")
    parser.add_argument("--model")
    args = parser.parse_args()
    print(select(args.source, args.target, args.seconds, args.direction, args.model))
