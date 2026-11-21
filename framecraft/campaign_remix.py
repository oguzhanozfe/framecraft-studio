"""Add music and a dry voice to an approved edit while retaining its original video stream."""

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import tempfile

from filelock import FileLock

from .campaign_motion import FPS, audit, render_soundtrack, run, target_frames
from .core import read, save
from .projects import archive


def approved_shot(picture_recipe, shot_number, offset, seconds, destination):
    """Reuse a timed excerpt of a previously audited edition with its source locked."""
    recipe = read(picture_recipe)
    parent = picture_recipe.parent
    audit_report = read(parent / "motion-assessment.json")
    with (parent / recipe["output"]).open("rb") as stream:
        picture_digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if not audit_report["passed"] or picture_digest != audit_report["sha256"]:
        raise ValueError("The selected edition has no matching approved frame audit")
    cue = audit_report["timeline"][shot_number]
    if (
        offset < 0
        or seconds <= 0
        or offset + seconds > (cue["to"] - cue["from"]) / FPS + 0.00001
    ):
        raise ValueError("The excerpt must stay within the approved shot")
    source = parent / cue["source"]
    with source.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    start = cue["source_start_seconds"] + offset
    return {
        "file": os.path.relpath(source, destination),
        "start": start,
        "seconds": seconds,
        "approved_span": {
            "edition_sha256": picture_digest,
            "sha256": digest,
            "start": start,
            "seconds": seconds,
        },
    }


def remix(picture_recipe, voice, voice_receipt, score, destination, *, asr=None):
    destination.mkdir(parents=True, exist_ok=True)
    with FileLock(str(destination / "remix.lock"), timeout=3600):
        old = read(picture_recipe)
        report = read(picture_recipe.parent / "motion-assessment.json")
        picture = picture_recipe.parent / old["output"]
        with picture.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if not report["passed"] or digest != report["sha256"]:
            raise ValueError("The picture has no matching approved frame audit")
        for cue in report["timeline"]:
            shutil.copyfile(
                picture_recipe.parent / cue["plate"], destination / cue["plate"]
            )
        spec = dict(old)
        spec["shots"] = []
        for shot in old["shots"]:
            shot = dict(shot)
            shot["file"] = os.path.relpath(
                picture_recipe.parent / shot["file"], destination
            )
            if shot.get("inset"):
                shot["inset"] = dict(shot["inset"])
                shot["inset"]["file"] = os.path.relpath(
                    picture_recipe.parent / shot["inset"]["file"], destination
                )
            spec["shots"].append(shot)
        if bool(voice) != bool(voice_receipt):
            raise ValueError("Supply both the voice and its receipt")
        for key in (
            "voice",
            "asr_model",
            "verify_voice_alignment",
            "spoken",
            "voice_provider",
            "voice_profile",
            "voice_cues",
        ):
            spec.pop(key, None)
        spec.update(
            silent=False,
            audio_mode="voice_music" if voice else "music",
            score=os.path.relpath(score, destination),
            output="master.mp4",
        )
        if voice:
            speech = read(voice_receipt)
            spec.update(
                voice=os.path.relpath(voice, destination),
                spoken=speech["text"],
                verify_voice_alignment=True,
                voice_provider=speech["provider"],
                voice_profile=speech.get("profile"),
            )
        if voice and asr:
            spec["asr_model"] = os.path.relpath(asr, destination)
        with tempfile.TemporaryDirectory(dir=destination) as scratch:
            tmp = Path(scratch)
            audio = tmp / "soundtrack.m4a"
            render_soundtrack(spec, destination, audio, target_frames(spec) / FPS)
            result = tmp / "master.mp4"
            run(
                [
                    "-i",
                    picture,
                    "-i",
                    audio,
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0",
                    "-c",
                    "copy",
                    "-movflags",
                    "+faststart",
                    result,
                ]
            )
            audited = audit(result, destination, report["timeline"], spec)
            if not audited["passed"]:
                raise ValueError("Sound edition QA: " + ", ".join(audited["failures"]))
            output = destination / spec["output"]
            result.replace(output)
            save(destination / "motion.json", spec)
            if spec.get("project"):
                archive(
                    spec["project"],
                    output,
                    stage="master",
                    report=destination / "motion-assessment.json",
                    ledger=destination / "motion-frames.jsonl.gz",
                )
        return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("picture_recipe", type=Path)
    parser.add_argument("voice", type=Path, nargs="?", help="Optional dry narration")
    parser.add_argument("voice_receipt", type=Path, nargs="?")
    parser.add_argument("score", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--asr", type=Path)
    args = parser.parse_args()
    print(remix(**vars(args)))
