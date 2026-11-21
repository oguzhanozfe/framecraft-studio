"""Build and verify a presenter's talking takes from their own-camera footage and voice."""

import argparse
import hashlib
import json
import os
import tempfile
import wave
from pathlib import Path

from filelock import FileLock

from .core import read, save


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def voice_bed(source, target, seconds, onset):
    with wave.open(str(source), "rb") as src:
        if src.getparams()[:3] != (1, 2, 24000):
            raise ValueError(f"Expected a 24 kHz mono PCM16 voice: {source}")
        samples = src.readframes(src.getnframes())
    size = 2 * round(24000 * seconds)
    start = 2 * round(24000 * onset)
    if start < 0 or not samples or start + len(samples) > size:
        raise ValueError(f"The voice at {onset:g}s does not fit its {seconds:g}s shot")
    data = bytearray(size)
    data[start : start + len(samples)] = samples
    with wave.open(str(target), "wb") as dst:
        dst.setparams((1, 2, 24000, 0, "NONE", ""))
        dst.writeframes(data)


def prepare_take(
    network, source, audio, output, detector, syncnet, *, start, seconds, batch=4
):
    from .campaign_sync import measure

    report = output.with_name(output.stem.removesuffix("-face") + "-sync.json")
    identity = {
        "source_sha256": digest(source),
        "audio_sha256": digest(audio),
        "source_start": start,
        "seconds": seconds,
    }
    with FileLock(str(output) + ".lock"):
        if output.is_file() and report.is_file():
            previous = read(report)
            if (
                previous.get("passed")
                and all(previous.get(k) == v for k, v in identity.items())
                and previous.get("video_sha256") == digest(output)
            ):
                return previous
        with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
            folder = Path(tmp)
            bed, video = folder / "voice.wav", folder / "talking.mp4"
            delay = 0
            for _ in range(3):
                onset = 0.25
                voice_bed(audio, bed, seconds, onset)
                network.run(
                    source,
                    bed,
                    video,
                    start,
                    0,
                    seconds,
                    batch=batch,
                    voice_delay_ms=delay,
                    speech_only=True,
                )
                check = measure(video, bed, detector, syncnet, seconds=seconds)
                offset = check.get("offset_ms", 0)
                if 40 <= abs(offset) <= 200 and check.get("confidence", 0) >= 0.55:
                    aligned = round(onset - offset / 1000, 3)
                    try:
                        voice_bed(audio, bed, seconds, aligned)
                    except ValueError:
                        pass
                    else:
                        trial = measure(video, bed, detector, syncnet, seconds=seconds)
                        if trial["passed"] and abs(trial["offset_ms"]) < abs(offset):
                            onset, check = (
                                aligned,
                                {**trial, "initial_offset_ms": offset},
                            )
                if check["passed"]:
                    record = {
                        **identity,
                        **check,
                        "voice_onset": onset,
                        "video_sha256": digest(video),
                    }
                    os.replace(video, output)
                    save(report, record)
                    return record
                if check.get("confidence", 0) < 0.55 or not offset:
                    break
                delay += offset
            raise ValueError(
                f"Rejected speaking take {audio.stem}: {json.dumps(check)}"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument("musetalk", type=Path)
    parser.add_argument("syncnet", type=Path)
    parser.add_argument("--product")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    from .campaign_dub import Dubber

    network = Dubber(args.weights, args.musetalk, args.device)
    root = args.catalog.resolve().parent
    for product in read(args.catalog)["products"]:
        if args.product and product["slug"] != args.product:
            continue
        speech = root / product["slug"] / "speech"
        for role, prefix, count in (
            ("opening", "hook", product["hooks"]),
            ("closing", "cta", 2),
        ):
            shot = product["presenter"][role]
            for i in range(count):
                key = f"{prefix}{i}"
                result = prepare_take(
                    network,
                    root / shot["file"],
                    speech / (key + ".wav"),
                    speech / (key + "-face.mp4"),
                    args.weights / "yunet.onnx",
                    args.syncnet,
                    start=shot.get("start", 0),
                    seconds=shot["seconds"],
                    batch=2 if args.device == "cuda-stream" else 4,
                )
                print(product["slug"], key, result["offset_ms"], flush=True)


if __name__ == "__main__":
    main()
