"""Render one person's phone recording across connected, independently generated takes."""

import argparse
import hashlib
import json
from pathlib import Path
import secrets

import cv2
from filelock import FileLock
from pydantic import BaseModel, Field, field_validator

from .core import read, save
from .director import CreativeIntent, ask
from .providers import generate


class CameraBeat(BaseModel):
    action: str = Field(min_length=30, max_length=800)
    headline: str = Field(min_length=2, max_length=50)
    narration: str = Field(min_length=10, max_length=160)

    @field_validator("narration")
    @classmethod
    def grounded(cls, line):
        from .motion_copy import unsupported_testimony

        if unsupported_testimony(line):
            raise ValueError(
                "Use an independently sourced fact or a preference instead of an invented test or universal claim"
            )
        return line


class CameraPlan(BaseModel):
    intent: CreativeIntent
    setting: str
    shots: list[CameraBeat] = Field(min_length=4, max_length=4)


def campaign(item, output, evidence, stage):
    from .campaign_motion import render
    from .campaign_voice import place_read, synthesize
    from .motion_campaign import add_score

    beats = [7, 8, 8, 7]
    brief = {k: item[k] for k in ("name", "audience", "angle", "proof")}
    brief["camera_setup"] = "direct_eye_level_v2"
    cache = output / "shots.json"
    if cache.is_file() and read(cache).get("phone_brief") == brief:
        plan = read(cache)
    else:
        story = ask(
            CameraPlan,
            "Direct ONE adult recording a vertical phone video. Start directly in the front lens at eye level, mid-reaction to the product's actual appeal. Keep the same operator, outfit, room and fixed furniture layout. Write four concrete English camera actions. For the middle beats show a plausible hand-level use and a second physical detail in that same environment. End with the creator doing the activity they wanted the product for. No mirrors, reflections of the phone, floor-level opening or aimless room tour. Describe compatible endpoints for adjacent camera moves. Four compact Turkish captions and four idiomatic Turkish voice-over lines, each 7 to 12 spoken words. Sell a specific desire using only documented facts. Ground each line in its shot. Write spoken numbers in Turkish.",
            brief,
            model_override=item.get("editor_model"),
        )
        plan = {**story.model_dump(), "phone_brief": brief}
        save(cache, plan)
    if stage == "prepare":
        return cache
    camera = {
        "setting": plan["setting"],
        "vision_model": item.get("vision_model"),
        "provider": item.get("render_model") or "h3",
        "shots": [
            {**s, "seconds": n, "layout": "caption"}
            for s, n in zip(plan["shots"], beats, strict=True)
        ],
    }
    if camera["provider"] == "auto":
        camera["provider"] = "h3"
    recipe = {
        "shots": film(camera, output),
        "phone_continuity": True,
        "silent": bool(item.get("silent")),
        "evidence": [evidence["url"]],
        "project": item.get("project") or item["name"],
        "output": "master.mp4",
    }
    if not recipe["silent"] and item.get("audio") != "music":
        lines = [s["narration"] for s in plan["shots"]]
        raw = output / "voice.wav"
        receipt = output / "voice.json"
        if not (
            raw.is_file()
            and receipt.is_file()
            and read(receipt).get("text") == " ".join(lines)
            and read(receipt).get("profile", "freya")
            == item.get("voice_profile", "freya")
        ):
            save(
                receipt,
                synthesize(
                    " ".join(lines),
                    raw,
                    segments=lines,
                    profile=item.get("voice_profile", "freya"),
                ),
            )
        voice = read(receipt)
        recipe["voice_cues"] = place_read(
            raw, output / "timed-voice.wav", voice["segments"], beats
        )
        recipe.update(
            voice="timed-voice.wav",
            spoken=voice["text"],
            voice_provider=voice["provider"],
            voice_profile=voice.get("profile", "freya"),
            verify_voice_alignment=True,
        )
        if item.get("asr_model"):
            recipe["asr_model"] = str(
                (output.parent.parent / item["asr_model"]).resolve()
            )
    if not recipe["silent"]:
        if item.get("audio") == "music":
            recipe["audio_mode"] = "music"
        add_score(item, output, recipe)
    save(output / "motion.json", recipe)
    return render(recipe, output)


def tail_anchor(path, seconds, target):
    camera = cv2.VideoCapture(str(path))
    try:
        fps = camera.get(cv2.CAP_PROP_FPS)
        camera.set(cv2.CAP_PROP_POS_FRAMES, max(0, round(seconds * fps) - 1))
        ok, frame = camera.read()
        if not ok or not cv2.imwrite(str(target), frame):
            raise ValueError(
                f"Could not read the outgoing phone frame from {path.name}"
            )
    finally:
        camera.release()


def film(plan, root):
    """The exact last used frame is the incoming camera pose for the next model call."""
    root.mkdir(exist_ok=True, parents=True)
    provider = plan.get("provider", "h3")
    shots = []
    previous = None
    for index, beat in enumerate(plan["shots"], 1):
        seconds = beat["seconds"]
        if not isinstance(seconds, int) or not 1 <= seconds <= 10:
            raise ValueError("Use integer camera beats of one to ten seconds")
        folder = root / f"phone-{index:02}"
        folder.mkdir(exist_ok=True)
        start = 1 / 24 if previous else 0
        anchor = None
        if previous:
            anchor = folder / "incoming.png"
            tail_anchor(
                root / previous["file"], previous["start"] + previous["seconds"], anchor
            )
        closing = None
        if provider == "h3" and shots and index == len(plan["shots"]):
            closing = folder / "return-to-creator.png"
            tail_anchor(root / shots[0]["file"], 0.5, closing)
        prompt = (
            "One camera, the adult creator's own vertical phone. Continuous real-time footage with imperfect wrist-held framing. "
            "The view moves only where that same phone physically goes. No second camera, jump cuts, graphic overlays or transitions. "
            "She records pictures for a voice-over later; mouth at rest. "
            + plan.get("setting", "")
            + " "
            + beat["action"]
        )
        signature = hashlib.sha256(
            json.dumps([provider, prompt, seconds]).encode()
            + (anchor.read_bytes() if anchor else b"")
            + (closing.read_bytes() if closing else b"")
        ).hexdigest()
        receipt, target = folder / "take.json", folder / "take.mp4"
        if not (
            target.is_file()
            and receipt.is_file()
            and read(receipt).get("signature") == signature
        ):
            seed = secrets.randbelow(2**31 - 1)
            tmp = folder / "take.partial.mp4"
            generate(
                provider,
                prompt,
                seed,
                "vertical",
                tmp,
                lambda x: print(x, flush=True),
                frames=seconds * 24 + 1,
                reference=anchor,
                end_reference=closing,
                front=bool(plan.get("priority")),
            )
            tmp.replace(target)
            save(
                receipt,
                {
                    "signature": signature,
                    "seed": seed,
                    "model": provider,
                    "prompt": prompt,
                    "incoming_frame_sha256": hashlib.sha256(
                        anchor.read_bytes()
                    ).hexdigest()
                    if anchor
                    else None,
                },
            )
        from .phone_review import repair

        target = repair(
            target,
            seconds,
            plan.get("setting", ""),
            incoming=anchor,
            outgoing=closing,
            model=plan.get("vision_model"),
            provider=provider,
            front=bool(plan.get("priority")),
            action=beat["action"],
        )
        previous = {
            "file": target.relative_to(root).as_posix(),
            "start": start,
            "seconds": seconds,
            **{
                k: beat[k]
                for k in ("layout", "headline", "detail", "credit")
                if k in beat
            },
        }
        shots.append(previous)
    return shots


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    root = args.plan.resolve().parent
    plan = read(args.plan)
    with FileLock(str(root / "phone.lock"), timeout=3600):
        shots = film(plan, root)
        save(
            root / "motion.json",
            {**plan["edit"], "phone_continuity": True, "shots": shots},
        )
