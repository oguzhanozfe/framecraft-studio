"""Quiet an on-camera cutaway using registered resting poses from the same take."""

import argparse
import math
import os
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from filelock import FileLock

from .campaign_experiments import fingerprint
from .campaign_face import (
    face_boxes,
    lip_mask,
    rest_crop,
    resting_references,
    speech_activity,
    track,
)
from .core import save


def render(source, output, *, start=0.0, seconds=None, voice=None):
    """Write a silent MP4 with the original framing, head movement and hand action."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output:
        raise ValueError("Use a separate output path for the listening take")
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output) + ".lock"):
        source_hash = fingerprint(source)
        cap = cv2.VideoCapture(str(source))
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if not cap.isOpened() or fps <= 0 or not math.isfinite(start) or start < 0:
                raise ValueError("Cannot read the requested source interval")
            first = round(start * fps)
            if seconds is not None and (not math.isfinite(seconds) or seconds <= 0):
                raise ValueError("Choose a positive duration for the listening take")
            count = total - first if seconds is None else round(seconds * fps)
            if count < 1 or first + count > total:
                raise ValueError("The requested interval exceeds the source footage")
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            mesh = track(source, first / fps, count)
            refs = resting_references(mesh)
            resting = np.ones(count, np.float32)
            if voice is not None:
                voice = Path(voice)
                voice_hash = fingerprint(voice)
                pcm = subprocess.check_output(
                    [
                        "ffmpeg",
                        "-nostdin",
                        "-v",
                        "error",
                        "-i",
                        str(voice),
                        "-vn",
                        "-ar",
                        "16000",
                        "-ac",
                        "1",
                        "-f",
                        "f32le",
                        "pipe:1",
                    ]
                )
                signal = np.frombuffer(pcm, dtype="<f4")
                if len(signal) / 16000 < (count - 1) / fps:
                    raise ValueError(
                        "Supply a timed dialogue WAV spanning the speaking shot"
                    )
                resting -= speech_activity(signal, count, fps)
                refs[resting == 0] = np.arange(count)[resting == 0]
            if len(refs) != count or np.any(refs < 0):
                raise ValueError("No registered resting expression in the same take")
            boxes = face_boxes(mesh, width, height)
            with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
                folder = Path(tmp)
                result = folder / "quiet.mp4"
                _encode(
                    cap,
                    folder,
                    result,
                    first,
                    fps,
                    width,
                    height,
                    mesh,
                    boxes,
                    refs,
                    resting,
                )
                if fingerprint(source) != source_hash:
                    raise ValueError("The input changed during generation")
                report = {
                    "source": str(source),
                    "source_sha256": source_hash,
                    "sha256": fingerprint(result),
                    "source_start": first / fps,
                    "frames": count,
                    "fps": fps,
                    "width": width,
                    "height": height,
                    "same_actor_resting_references": len(set(refs)),
                }
                if voice is not None:
                    if fingerprint(voice) != voice_hash:
                        raise ValueError("Dialogue changed during pause correction")
                    report.update(
                        timed_voice_sha256=voice_hash,
                        preserved_speaking_frames=int(np.count_nonzero(resting == 0)),
                    )
                os.replace(result, output)
                save(output.with_suffix(".source.json"), report)
                return report
        finally:
            cap.release()


def _encode(cap, folder, result, first, fps, width, height, mesh, boxes, refs, resting):
    donors = {ref for i, ref in enumerate(refs) if i != ref and resting[i] > 0}
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    for index in range(len(refs)):
        if not cap.grab():
            raise ValueError("Could not decode the reference interval")
        if index in donors:
            ok, frame = cap.retrieve()
            if not ok:
                raise ValueError("Could not read a resting reference")
            np.save(folder / f"{index}.npy", frame)

    # Use only a few decoded reference frames in RAM even for full-resolution footage.
    @lru_cache(maxsize=4)
    def reference_frame(index):
        return np.load(folder / f"{index}.npy", allow_pickle=False)

    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    command = [
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "15",
        "-threads",
        "3",
        "-pix_fmt",
        "yuv420p",
        str(result),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for index, (box, ref) in enumerate(zip(boxes, refs, strict=True)):
            ok, frame = cap.read()
            if not ok:
                raise ValueError("Could not decode the complete source take")
            if ref != index and resting[index] > 0:
                x, y, right, bottom = box
                surface = frame[y:bottom, x:right]
                reference = rest_crop(
                    {index: frame, ref: reference_frame(ref)}, mesh, index, ref, box
                )
                mask = lip_mask(mesh[index], box) * resting[index]
                surface[:] = np.rint(surface * (1 - mask) + reference * mask).astype(
                    np.uint8
                )
            encoder.stdin.write(frame.tobytes())
        encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError("Listening take encoding failed")
    finally:
        if encoder.poll() is None:
            encoder.kill()
            encoder.wait()
        if not encoder.stdin.closed:
            encoder.stdin.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--seconds", type=float)
    parser.add_argument(
        "--voice",
        type=Path,
        help="Keep the existing performance during speech in this timed WAV",
    )
    args = parser.parse_args()
    render(
        args.source,
        args.output,
        start=args.start,
        seconds=args.seconds,
        voice=args.voice,
    )
