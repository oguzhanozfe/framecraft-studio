"""Inspect camera orientation and operator continuity, then repair only a rejected take."""

import hashlib
from pathlib import Path
import secrets

import cv2
import numpy as np
from pydantic import BaseModel, Field

from .core import read, save
from .director import ask
from .providers import generate


class PhoneCheck(BaseModel):
    observation: str
    sideways_tiles: list[int] = Field(default_factory=list)
    impossible_anatomy_tiles: list[int] = Field(default_factory=list)
    plausible_phone_camera: bool


def inspect(path, model=None):
    cap = cv2.VideoCapture(str(path))
    try:
        tiles = []
        for i, fraction in enumerate(np.linspace(0.03, 0.95, 8), 1):
            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                round((cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1) * fraction),
            )
            ok, image = cap.read()
            if not ok:
                raise ValueError("Cannot review the recorded phone angle")
            small = cv2.resize(image, (176, 312), interpolation=cv2.INTER_AREA)
            cv2.putText(
                small, str(i), (6, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 255, 255), 2
            )
            tiles.append(small)
    finally:
        cap.release()
    sheet = np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])])
    jpg = cv2.imencode(".jpg", sheet)[1].tobytes()
    dest = path.with_suffix(".camera.json")
    signature = hashlib.sha256(jpg).hexdigest()
    if dest.is_file() and read(dest).get("signature") == signature:
        return read(dest)
    result = ask(
        PhoneCheck,
        "Review eight sequential numbered frames for camera continuity. Describe what the operator actually filmed. Give numbers only for tiles rolled at least 55 degrees onto their side (use the orientation of the person, desk and door). Give separate tile numbers for CLEAR duplicate faces on one body, an impossible phone grip, impossible arm/body, or conspicuously morphing objects. A real mirror reflection consistent with a single adult is allowed. A slight tilt, ordinary camera shake and a phone or subject that moves out of view are normal. Decide whether a person holding one phone could physically have recorded this view.",
        {},
        jpg,
        model_override=model,
    )
    value = {
        **result.model_dump(),
        "signature": signature,
        "passed": not result.sideways_tiles
        and not result.impossible_anatomy_tiles
        and result.plausible_phone_camera,
    }
    save(dest, value)
    return value


def repair(
    path,
    seconds,
    setting,
    *,
    incoming=None,
    outgoing=None,
    model=None,
    front=False,
    provider="h3",
    action="",
):
    path = Path(path)
    report = inspect(path, model)
    if report["passed"]:
        return path
    for attempt in range(1, 3):
        target = path.with_name(f"{path.stem}-repair-{attempt}.mp4")
        prompt = f"{setting} {action}. One physically continuous upright portrait phone recording. Keep the doorframe vertical, image level and just one adult operator; no camera roll or clones. Make a natural small step to reframe at the wrist. Preserve the room and camera direction. {report['observation'][:240]} was the rejected take; correct only its visible filming defects."
        signature = hashlib.sha256(
            prompt.encode()
            + b"".join(Path(a).read_bytes() for a in (incoming, outgoing) if a)
        ).hexdigest()
        receipt = target.with_suffix(".json")
        if not (
            target.is_file()
            and receipt.is_file()
            and read(receipt).get("signature") == signature
        ):
            seed = secrets.randbelow(2**31 - 1)
            tmp = target.with_suffix(".partial.mp4")
            generate(
                provider,
                prompt,
                seed,
                "vertical",
                tmp,
                lambda s: print(s, flush=True),
                frames=seconds * 24 + 1,
                reference=incoming,
                end_reference=outgoing if provider == "h3" else None,
                front=front,
            )
            tmp.replace(target)
            save(receipt, {"seed": seed, "signature": signature, "prompt": prompt})
        report = inspect(target, model)
        if report["passed"]:
            return target
    raise ValueError(
        f"Phone camera QA rejected {path.parent.name}: {report['observation']}"
    )
