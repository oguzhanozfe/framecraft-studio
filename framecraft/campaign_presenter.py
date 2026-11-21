"""Keep a single locally generated presenter through distinct camera setups."""

import argparse
from pathlib import Path
import secrets
import cv2
import numpy as np
from .core import read, save
from .providers import generate


def cast_frames(video, target, detector_path):
    cap = cv2.VideoCapture(str(video))
    detector = cv2.FaceDetectorYN.create(str(detector_path), "", (320, 320), 0.72)
    choices = []
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for f in np.linspace(0.1, 0.88, 12):
            cap.set(cv2.CAP_PROP_POS_FRAMES, round((total - 1) * f))
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            detector.setInputSize((w, h))
            _, dets = detector.detect(frame)
            if dets is None or len(dets) != 1:
                continue
            x, y, bw, bh = map(int, dets[0, :4])
            face = frame[max(0, y) : min(h, y + bh), max(0, x) : min(w, x + bw)]
            if min(face.shape[:2]) < 45:
                continue
            sharp = cv2.Laplacian(
                cv2.cvtColor(face, cv2.COLOR_BGR2GRAY), cv2.CV_64F
            ).var()
            choices.append((sharp * float(dets[0, -1]), f, frame))
    finally:
        cap.release()
    if not choices:
        raise ValueError("No single clear presenter face in first take")
    choices.sort(key=lambda e: -e[0])
    cv2.imwrite(str(target), choices[0][2])
    return choices[0][1]


def continue_cast(plan, folder, detector):
    root = folder / plan["opening"]
    anchor = folder / "cast-reference.png"
    position = cast_frames(root, anchor, detector)
    for s in plan["shots"]:
        target = folder / s["file"]
        receipt = target.with_suffix(".json")
        if (
            receipt.exists()
            and target.is_file()
            and read(receipt).get("prompt") == s["prompt"]
        ):
            continue
        seed = secrets.randbelow(2**31 - 1)
        tmp = target.with_suffix(".partial.mp4")
        generate(
            "h3",
            s["prompt"],
            seed,
            "vertical",
            tmp,
            lambda x: print(x, flush=True),
            frames=s["seconds"] * 24 + 1,
            reference=anchor,
        )
        tmp.replace(target)
        save(
            receipt,
            {
                "provider": "h3",
                "seed": seed,
                "prompt": s["prompt"],
                "reference": anchor.name,
                "source": root.name,
                "reference_position": position,
            },
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("plan", type=Path)
    ap.add_argument("detector", type=Path)
    args = ap.parse_args()
    continue_cast(read(args.plan), args.plan.parent, args.detector)
