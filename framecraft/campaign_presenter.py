"""Keep a single locally generated presenter through distinct camera setups."""

import argparse
import hashlib
from pathlib import Path
import secrets
import cv2
import numpy as np
from pydantic import BaseModel, Field
from .core import read, save
from .director import ask
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


class CastingChoice(BaseModel):
    row: int = Field(ge=1)
    observation: str = Field(min_length=10)
    visible_defects: list[str]


def choose_opening(plan, folder):
    candidates = list(plan["candidates"].items())
    strips = []
    for idx, (_, file) in enumerate(candidates, 1):
        cap = cv2.VideoCapture(str(folder / file))
        tiles = []
        try:
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for part in (0.08, 0.36, 0.64, 0.90):
                cap.set(cv2.CAP_PROP_POS_FRAMES, round((length - 1) * part))
                ok, img = cap.read()
                if not ok:
                    raise ValueError(f"Unreadable candidate: {file}")
                tiles.append(cv2.resize(img, (216, 384), interpolation=cv2.INTER_AREA))
        finally:
            cap.release()
        strip = np.hstack(tiles)
        cv2.rectangle(strip, (0, 0), (63, 35), (0, 0, 0), -1)
        cv2.putText(
            strip,
            str(idx),
            (10, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
        )
        strips.append(strip)
    board = np.vstack(strips)
    cv2.imwrite(str(folder / "casting.jpg"), board)
    response = ask(
        CastingChoice,
        "Judge the actual contact sheet. Each numbered row is an anonymous local video model, showing four chronological frames. Choose the row closest to the casting brief. Base the decision on natural skin, physical plausibility, candid action, wardrobe and identity continuity. Cite what you can see, and identify visible defects even in the winner.",
        {"brief": plan["casting_brief"], "row_count": len(candidates)},
        cv2.imencode(".jpg", board)[1].tobytes(),
        model_override=plan.get("vision_model"),
    )
    if response.row > len(candidates):
        raise ValueError("Casting model selected a nonexistent row")
    provider, opening = candidates[response.row - 1]
    save(
        folder / "casting.json",
        {"provider": provider, "opening": opening, **response.model_dump()},
    )
    return {**plan, "provider": provider, "opening": opening}


def continue_cast(plan, folder, detector):
    if plan.get("candidates"):
        plan = choose_opening(plan, folder)
    root = folder / plan["opening"]
    anchor = folder / "cast-reference.png"
    position = cast_frames(root, anchor, detector)
    anchor_hash = hashlib.sha256(anchor.read_bytes()).hexdigest()
    for s in plan["shots"]:
        provider = s.get("provider", plan.get("provider", "h3"))
        target = folder / s["file"]
        receipt = target.with_suffix(".json")
        if (
            receipt.exists()
            and target.is_file()
            and read(receipt).get("prompt") == s["prompt"]
            and read(receipt).get("provider") == provider
            and read(receipt).get("reference_sha256") == anchor_hash
        ):
            continue
        seed = secrets.randbelow(2**31 - 1)
        tmp = target.with_suffix(".partial.mp4")
        generate(
            provider,
            s["prompt"],
            seed,
            "vertical",
            tmp,
            lambda x: print(x, flush=True),
            frames=s["seconds"] * 24 + 1,
            reference=anchor,
            front=bool(plan.get("priority", False)),
        )
        tmp.replace(target)
        save(
            receipt,
            {
                "provider": provider,
                "seed": seed,
                "prompt": s["prompt"],
                "reference": anchor.name,
                "reference_sha256": anchor_hash,
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
