"""Inspect camera orientation and operator continuity, then repair only a rejected take."""

import hashlib
from pathlib import Path
import secrets
import subprocess

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
    continuity_break_tiles: list[int] = Field(
        description="Tiles that show a subject or background morph, impossible scene switch, or unmotivated internal cut. Empty when the view moves continuously."
    )
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
    signature = hashlib.sha256(b"camera-continuity-2" + jpg).hexdigest()
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
        and not result.continuity_break_tiles
        and result.plausible_phone_camera,
    }
    save(dest, value)
    return value


def patch_roll(path, report, setting, model, front=False):
    """Reshoot an interior roll between two upright frames, keeping the other action."""
    bad = sorted(
        set(
            report["sideways_tiles"]
            + report["impossible_anatomy_tiles"]
            + report.get("continuity_break_tiles", [])
        )
    )
    if not bad or min(bad) < 3 or max(bad) > 6:
        return None
    cap = cv2.VideoCapture(str(path))
    try:
        fps = round(cap.get(cv2.CAP_PROP_FPS))
        total = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps != 24 or total < 48:
            return None
        positions = np.rint(np.linspace(0.03, 0.95, 8) * (total - 1)).astype(int)
        left, right = int(positions[min(bad) - 3]), int(positions[max(bad) + 1])
        anchors = []
        for pos in (left, right):
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ok, image = cap.read()
            target = path.with_name(f"{path.stem}-anchor-{pos}.png")
            if not ok or not cv2.imwrite(str(target), image):
                raise ValueError("Missing anchor at a reported camera roll")
            anchors.append(target)
    finally:
        cap.release()
    output = path.with_name(path.stem + "-joined.mp4")
    receipt = output.with_suffix(".json")
    if not (
        output.is_file()
        and receipt.is_file()
        and read(receipt).get("input") == report["signature"]
    ):
        replacement = path.with_name(path.stem + "-bridge.mp4")
        seed = secrets.randbelow(2**31 - 1)
        prompt = (
            setting
            + " Match the two supplied frames with a single steady upright portrait phone recording. "
            "A modest sideways shift of the operator's wrist, a breath, a change of gaze. "
            "Keep gravity vertical, geometry rigid and the horizon level through the entire move. One person, one handset."
        )
        generate(
            "h3",
            prompt,
            seed,
            "vertical",
            replacement,
            lambda s: print(s, flush=True),
            frames=right - left + 1,
            reference=anchors[0],
            end_reference=anchors[1],
            front=front,
        )
        bridge = cv2.VideoCapture(str(replacement))
        try:
            size = int(bridge.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            bridge.release()
        if size < 2:
            raise ValueError("Empty phone-camera repair")
        # Address the generated frames by normalized time; H3 rounds native clip lengths.
        count = right - left + 1
        filters = (
            f"[0:v]split=2[a][b];[a]trim=end_frame={left},setpts=PTS-STARTPTS[before];"
            f"[1:v]setpts=(PTS-STARTPTS)*{count - 1}/{size - 1},fps={fps}:round=near,tpad=stop_mode=clone:stop=1,trim=end_frame={count},setpts=PTS-STARTPTS[fix];"
            f"[b]trim=start_frame={right + 1},setpts=PTS-STARTPTS[after];"
            f"[before][fix][after]concat=n=3:v=1:a=0,setsar=1[v]"
        )
        tmp = output.with_suffix(".partial.mp4")
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-i",
                str(path),
                "-i",
                str(replacement),
                "-filter_complex",
                filters,
                "-map",
                "[v]",
                "-an",
                "-frames:v",
                str(total),
                "-c:v",
                "libx264",
                "-crf",
                "15",
                "-preset",
                "veryfast",
                str(tmp),
            ],
            check=True,
        )
        tmp.replace(output)
        save(
            receipt,
            {
                "input": report["signature"],
                "replaced_frames": [left, right],
                "seed": seed,
            },
        )
    return output if inspect(output, model)["passed"] else None


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
        if provider == "h3":
            fixed = patch_roll(
                path if attempt == 1 else target, report, setting, model, front
            )
            if fixed:
                return fixed
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
