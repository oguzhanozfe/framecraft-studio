"""Check the mouth against the actual speech using a local pretrained SyncNet."""

import argparse
import json
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


def _version(path):
    path = Path(path).resolve()
    stat = path.stat()
    return path, stat.st_size, stat.st_mtime_ns


def load_model(model_dir):
    return _load_model(_version(Path(model_dir) / "syncnet_v2.model"))


@lru_cache(maxsize=1)
def _load_model(weights):
    import torch

    model_dir = weights[0].parent
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))
    from SyncNetModel import S

    net = S()
    net.load_state_dict(
        torch.load(
            model_dir / "syncnet_v2.model", map_location="cpu", weights_only=True
        )
    )
    return net.eval()


def tracked_mouths(video, detector_path, start, seconds):
    # Timing and same-speaker checks inspect the very same decoded frames.
    return _tracked_mouths(_version(video), _version(detector_path), start, seconds)


@lru_cache(maxsize=1)
def _tracked_mouths(video_version, detector_version, start, seconds):
    video, detector_path = video_version[0], detector_version[0]
    from scipy.ndimage import median_filter

    capture = cv2.VideoCapture(str(video))
    actual_rate = capture.get(cv2.CAP_PROP_FPS)
    if not capture.isOpened() or actual_rate <= 0:
        capture.release()
        raise ValueError(f"Cannot read video frame rate: {video}")
    duration = capture.get(cv2.CAP_PROP_FRAME_COUNT) / actual_rate
    seconds = min(
        seconds if seconds is not None else duration - start, duration - start
    )
    count = int(seconds * 25)
    if start < 0 or count < 15:
        capture.release()
        raise ValueError("A speech interval of at least 0.6 seconds is required")
    detector = cv2.FaceDetectorYN.create(str(detector_path), "", (320, 320), 0.65)
    frames, boxes = [], []
    source_frame = round(start * actual_rate)
    capture.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
    decoded = source_frame - 1
    try:
        for i in range(count):
            target = round((start + i / 25) * actual_rate)
            while decoded < target:
                ok, im = capture.read()
                if not ok:
                    raise ValueError("Cannot decode the visible speech interval")
                decoded += 1
            h, w = im.shape[:2]
            scale = min(1, 448 / w, 800 / h)
            frame = cv2.resize(
                im, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA
            )
            frames.append(frame)
            detector.setInputSize((frame.shape[1], frame.shape[0]))
            _, faces = detector.detect(frame)
            if faces is None or len(faces) != 1:
                boxes.append(None)
            else:
                x, y, w, h = faces[0, :4]
                boxes.append((x + w / 2, y + h / 2, max(w, h) / 2))
    finally:
        capture.release()
    visible = [i for i, v in enumerate(boxes) if v is not None]
    if len(visible) < 0.82 * count:
        return None, len(visible), count, seconds
    track = np.stack(
        [
            np.interp(np.arange(count), visible, [boxes[i][k] for i in visible])
            for k in range(3)
        ],
        axis=1,
    )
    track = median_filter(track, size=(min(13, count // 2 * 2 + 1), 1), mode="nearest")
    crops = []
    for frame, (x, y, r) in zip(frames, track, strict=True):
        pad = int(2 * r) + 4
        padded = cv2.copyMakeBorder(
            frame, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(110, 110, 110)
        )
        x += pad
        y += pad
        face = padded[
            int(y - r) : int(y + 1.8 * r), int(x - 1.4 * r) : int(x + 1.4 * r)
        ]
        crops.append(cv2.resize(face, (224, 224)))
    return crops, len(visible), count, seconds


def read_pcm(audio, audio_start, seconds):
    if audio_start < 0:
        raise ValueError("Audio start must be nonnegative")
    raw = subprocess.check_output(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-ss",
            str(audio_start),
            "-i",
            str(audio),
            "-t",
            str(seconds),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-f",
            "s16le",
            "-",
        ]
    )
    return np.frombuffer(raw, dtype="<i2")


def visual_features(net, crops, windows):
    import torch

    images = torch.from_numpy(np.stack(crops)).permute(3, 0, 1, 2)
    features = []
    with torch.inference_mode():
        for first in range(0, windows, 8):
            indices = range(first, min(first + 8, windows))
            features.append(
                net.forward_lip(
                    torch.stack([images[:, i : i + 5] for i in indices]).float()
                )
            )
    return torch.cat(features)


def acoustic_features(net, pcm, max_windows):
    import python_speech_features
    import torch

    mfcc = python_speech_features.mfcc(pcm, 16000).T
    windows = min(max_windows, (mfcc.shape[1] - 20) // 4 + 1)
    if windows < 10:
        raise ValueError("Too little speech for audiovisual measurement")
    speech = torch.tensor(mfcc, dtype=torch.float32)[None]
    features = []
    with torch.inference_mode():
        for first in range(0, windows, 8):
            indices = range(first, min(first + 8, windows))
            features.append(
                net.forward_aud(
                    torch.stack([speech[:, :, 4 * i : 4 * i + 20] for i in indices])
                )
            )
    return torch.cat(features)


def observed_features(video, audio, detector, model_dir, start, seconds, audio_start):
    """Encode one exact interval once for timing and counterfactual comparisons."""
    return _observed_features(
        _version(video),
        _version(audio),
        _version(detector),
        _version(Path(model_dir) / "syncnet_v2.model"),
        start,
        seconds,
        audio_start,
    )


@lru_cache(maxsize=1)
def _observed_features(video, audio, detector, weights, start, seconds, audio_start):
    crops, visible, count, duration = tracked_mouths(
        video[0], detector[0], start, seconds
    )
    if crops is None:
        return None, None, None, visible, count, duration
    net = load_model(weights[0].parent)
    pcm = read_pcm(audio[0], audio_start, duration)
    sound = acoustic_features(net, pcm, len(crops) - 4)
    return visual_features(net, crops, len(sound)), sound, pcm, visible, count, duration


def measure(
    video, audio, detector_path, model_dir, start=0.0, seconds=None, audio_start=0.0
):
    import torch

    vis, sound, pcm, visible, count, seconds = observed_features(
        video, audio, detector_path, model_dir, start, seconds, audio_start
    )
    if vis is None:
        return {
            "passed": False,
            "reason": "face_track",
            "detected": visible,
            "frames": count,
        }
    windows = len(sound)
    # Compare the same speaking windows for every lag, excluding silence and padded edges.
    radius = min(8, (windows - 8) // 2)
    candidates = np.arange(radius, windows - radius)
    active = [
        i
        for i in candidates
        if np.mean((pcm[i * 640 : (i + 5) * 640].astype(float) / 32768) ** 2) > 0.00008
    ]
    if len(active) < 5:
        return {
            "passed": False,
            "reason": "insufficient_voiced_frames",
            "voiced_windows": len(active),
        }
    dist = [
        float(
            torch.linalg.vector_norm(
                vis[active] - sound[np.asarray(active) + lag], dim=1
            ).mean()
        )
        for lag in range(-radius, radius + 1)
    ]
    offset = int(np.argmin(dist)) - radius
    confidence = float(np.median(dist) - min(dist))
    return {
        "model": "SyncNet v2",
        "passed": abs(offset) <= 2 and confidence >= 0.55,
        "frame_rate": 25,
        "offset_ms": offset * 40,
        "confidence": round(confidence, 3),
        "voiced_windows": len(active),
        "face_frames": visible,
        "frames": count,
        "distances": [round(d, 3) for d in dist],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("video", "audio", "detector", "model_dir", "report"):
        parser.add_argument(key, type=Path)
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--audio-start", type=float, default=0)
    parser.add_argument("--seconds", type=float)
    args = parser.parse_args()
    result = measure(
        args.video,
        args.audio,
        args.detector,
        args.model_dir,
        args.start,
        args.seconds,
        args.audio_start,
    )
    args.report.write_text(json.dumps(result, indent=2), encoding="utf8")
    print(result, flush=True)
    raise SystemExit(0 if result["passed"] else 2)
