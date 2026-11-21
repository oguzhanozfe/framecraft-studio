"""Compare the decoded picture and mixed audio preceding the experimental closing."""

import argparse
import subprocess
from pathlib import Path

import numpy as np

from .campaign_edit import fingerprint
from .core import read, save


def decode(path, stream, args):
    return subprocess.check_output(
        [
            "ffmpeg",
            "-v",
            "error",
            "-threads",
            "2",
            "-i",
            str(path),
            "-map",
            f"0:{stream}:0",
            *args,
            "pipe:1",
        ]
    )


def picture(path):
    raw = decode(
        path,
        "v",
        [
            "-vf",
            "scale=80:142:flags=area",
            "-pix_fmt",
            "gray",
            "-threads",
            "1",
            "-f",
            "rawvideo",
        ],
    )
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, 80 * 142)


def sound(path):
    data = decode(path, "a", ["-ac", "1", "-ar", "8000", "-f", "f32le"])
    return np.frombuffer(data, dtype="<f4")


def compare(first, second, closing_frame, output, fps=30):
    hashes = [fingerprint(path) for path in (first, second)]
    if hashes[0] == hashes[1]:
        raise ValueError("The two editions are the same file")
    if output.exists():
        cached = read(output)
        if (
            cached.get("version") == 1
            and cached.get("sha256") == hashes
            and cached.get("closing_frame") == closing_frame
            and cached.get("fps") == fps
        ):
            return cached
    a, b = picture(first), picture(second)
    if a.shape != b.shape or not 0 < closing_frame < len(a) or fps <= 0:
        raise ValueError(
            "The edits must contain equally many frames and a closing interval"
        )
    difference = np.abs(a.astype(np.int16) - b).mean(axis=1)
    prefix = difference[:closing_frame]
    samples = round(8000 * closing_frame / fps)
    x, y = sound(first)[:samples], sound(second)[:samples]
    if len(x) != samples or len(y) != samples:
        raise ValueError("Audio ends before the closing frame")
    alevel, blevel = np.linalg.norm(x), np.linalg.norm(y)
    if min(alevel, blevel) < 0.01:
        raise ValueError("The paired audio has no measurable signal")
    correlation = float(np.dot(x, y) / (alevel * blevel))
    gain_db = float(20 * np.log10(alevel / blevel))
    closing = difference[closing_frame:]
    passed = bool(
        prefix.max() < 1.5
        and correlation >= 0.985
        and abs(gain_db) <= 1.5
        and closing.max() > 0.05
    )
    result = {
        "version": 1,
        "passed": passed,
        "sha256": hashes,
        "closing_frame": closing_frame,
        "fps": fps,
        "frames_compared": len(a),
        "prefix_frames": len(prefix),
        "prefix_max_luma_error_255": round(float(prefix.max()), 4),
        "prefix_mixed_audio_correlation": round(correlation, 6),
        "prefix_audio_level_difference_db": round(gain_db, 3),
        "closing_mean_luma_change_255": round(
            float(difference[closing_frame:].mean()), 4
        ),
    }
    save(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("closing_frame", type=int)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = compare(args.first, args.second, args.closing_frame, args.output)
    print(report)
    raise SystemExit(0 if report["passed"] else 2)
