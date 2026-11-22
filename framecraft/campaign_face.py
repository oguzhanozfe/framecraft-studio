"""Track and register a presenter's real face for speech and listening shots."""

import argparse
import os
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


def landmark_runtime():
    home = Path(os.environ.get("FRAMECRAFT_HOME", Path.home() / ".framecraft"))
    env = home / "runtimes" / "face-landmarks"
    executable = env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    python = os.environ.get("FRAMECRAFT_FACE_PYTHON", str(executable))
    model = Path(
        os.environ.get(
            "FRAMECRAFT_FACE_MODEL",
            home / "models/face-landmarker/face_landmarker.task",
        )
    )
    return python if Path(python).is_file() else sys.executable, model


def extract(source, model, start, count):
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=2,
    )
    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    result = []
    try:
        with vision.FaceLandmarker.create_from_options(options) as detector:
            for i in range(count):
                ok, frame = cap.read()
                if not ok:
                    raise ValueError("Presenter ends before the speech shot")
                h, w = frame.shape[:2]
                scale = min(1.0, 800 / max(h, w))
                if scale < 1:
                    frame = cv2.resize(
                        frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
                    )
                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                )
                faces = detector.detect_for_video(
                    image, round(i * 1000 / fps)
                ).face_landmarks
                if len(faces) > 1:
                    raise ValueError("Multiple faces in a speaking shot")
                result.append(
                    np.array(
                        [(landmark.x * w, landmark.y * h) for landmark in faces[0]],
                        np.float32,
                    )
                    if faces
                    else None
                )
    finally:
        cap.release()
    good = [i for i, face in enumerate(result) if face is not None]
    if len(good) < count * 0.9 or max(np.diff([-1, *good, count])) > max(
        4, round(fps * 0.20)
    ):
        raise ValueError("Incomplete facial landmark track in speaking shot")
    good_faces = np.stack([result[i] for i in good])
    for i, face in enumerate(result):
        if face is None:
            right = int(np.searchsorted(good, i))
            before, after = max(0, right - 1), min(right, len(good) - 1)
            t = (i - good[before]) / max(1, good[after] - good[before])
            result[i] = good_faces[before] + t * (
                good_faces[after] - good_faces[before]
            )
    return np.stack(result)


def track(source, start, count):
    python, model = landmark_runtime()
    if not model.is_file():
        raise FileNotFoundError(
            f"Install the MediaPipe Face Landmarker task or set FRAMECRAFT_FACE_MODEL: {model}"
        )
    source = Path(source).resolve()
    stat = source.stat()
    version = (stat.st_size, stat.st_mtime_ns, model.stat().st_mtime_ns)
    return _track_cached(source, start, count, python, model, version).copy()


@lru_cache(maxsize=12)
def _track_cached(source, start, count, python, model, version):
    with tempfile.TemporaryDirectory() as temp:
        target = Path(temp) / "track.npy"
        subprocess.run(
            [
                python,
                str(Path(__file__).resolve()),
                str(source),
                str(model),
                str(target),
                str(start),
                str(count),
            ],
            check=True,
        )
        return np.load(target, allow_pickle=False)


# Contour along the cheeks and under the chin. Keep hair, nose bridge and eyes
# from the original phone shot.
LOWER_FACE = [
    98,
    116,
    123,
    147,
    172,
    136,
    150,
    149,
    176,
    148,
    152,
    377,
    400,
    378,
    379,
    365,
    397,
    376,
    352,
    345,
    327,
]
POSE = [33, 263, 168, 1, 98, 327, 234, 454]


def face_boxes(mesh, width, height):
    oval = mesh[:, [234, 454, 93, 323, 132, 361, 172, 397, 152, 10]]
    left = oval[:, :, 0].min(axis=1)
    right = oval[:, :, 0].max(axis=1)
    chin = mesh[:, 152, 1]
    bridge = mesh[:, 6, 1]
    span = right - left
    box = np.stack(
        [
            left - 0.02 * span,
            2 * bridge - chin,
            right + 0.02 * span,
            chin + 0.06 * span,
        ],
        axis=1,
    )
    box = np.stack(
        [np.median(box[max(0, i - 1) : i + 2], axis=0) for i in range(len(box))]
    )
    box = np.rint(np.clip(box, [0, 0, 0, 0], [width, height, width, height])).astype(
        int
    )
    if np.min(box[:, 2:] - box[:, :2]) < 28:
        raise ValueError("Presenter is too small to dub")
    return box


def lower_mask(face, box):
    x, y, r, b = box
    mask = np.zeros((b - y, r - x), np.float32)
    cv2.fillConvexPoly(
        mask, cv2.convexHull(np.rint(face[LOWER_FACE] - [x, y]).astype(np.int32)), 1
    )
    return cv2.GaussianBlur(mask, (0, 0), max(1.0, (r - x) * 0.029))[:, :, None]


def lip_mask(face, box):
    x, y, r, b = box
    mouth = np.rint(
        face[[61, 40, 37, 0, 267, 270, 291, 321, 314, 17, 84, 91]] - [x, y]
    ).astype(np.int32)
    eyes = np.linalg.norm(face[33] - face[263])
    mask = np.zeros((b - y, r - x), np.float32)
    cv2.fillConvexPoly(mask, cv2.convexHull(mouth), 1)
    margin = max(2, round(eyes * 0.055))
    mask = cv2.dilate(
        mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1)),
    )
    return cv2.GaussianBlur(mask, (0, 0), max(0.9, eyes * 0.030))[:, :, None]


def mouth_transform(donor, target):
    # The two corners register scale, rotation and centre. An affine fit on a
    # turning head can turn one cheek and hair into a second mouth on screen.
    a, b = donor[291] - donor[61], target[291] - target[61]
    den = float(np.dot(a, a))
    if den < 25:
        return None
    cosine = float(a @ b) / den
    sine = float(a[0] * b[1] - a[1] * b[0]) / den
    scale = np.hypot(cosine, sine)
    if not 0.68 <= scale <= 1.45 or cosine / scale < 0.88:
        return None
    linear = np.array([[cosine, -sine], [sine, cosine]], np.float32)
    center = (target[61] + target[291]) / 2 - linear @ ((donor[61] + donor[291]) / 2)
    return np.column_stack([linear, center])


def resting_references(mesh):
    pose = mesh[:, POSE]
    scale = np.maximum(1, np.linalg.norm(mesh[:, 33] - mesh[:, 263], axis=1))
    norm = (pose - pose.mean(axis=1, keepdims=True)) / scale[:, None, None]
    width = np.linalg.norm(mesh[:, 61] - mesh[:, 291], axis=1)
    opening = np.linalg.norm(mesh[:, 13] - mesh[:, 14], axis=1) / np.maximum(width, 1)
    yaw = (mesh[:, 1, 0] - (mesh[:, 33, 0] + mesh[:, 263, 0]) / 2) / scale
    eligible = np.flatnonzero(
        opening <= min(0.085, max(0.025, np.percentile(opening, 30)))
    )
    output = np.full(len(mesh), -1, int)
    previous = -1
    for i in range(len(mesh)):
        if not len(eligible):
            continue
        delta = np.sqrt(((norm[eligible] - norm[i]) ** 2).mean(axis=(1, 2)))
        good = (delta < 0.075) & (abs(yaw[eligible] - yaw[i]) < 0.13)
        candidates = eligible[good]
        if not len(candidates):
            continue
        cost = (
            delta[good] + 0.18 * opening[candidates] + 0.00028 * np.abs(candidates - i)
        )
        if previous >= 0:
            cost += 0.001 * np.minimum(abs(candidates - previous - 1), 9)
        for rank in np.argsort(cost):
            j = candidates[rank]
            if mouth_transform(mesh[j], mesh[i]) is not None:
                output[i] = previous = j
                break
    return output


def rest_crop(frames, mesh, index, reference, box):
    x, y, r, b = box
    if reference < 0 or reference == index:
        return frames[index][y:b, x:r]
    transform = mouth_transform(mesh[reference], mesh[index])
    if transform is None:
        return frames[index][y:b, x:r]
    transform[:, 2] -= [x, y]
    return cv2.warpAffine(
        frames[reference],
        transform,
        (r - x, b - y),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def speech_activity(signal, count, fps):
    # Keep unvoiced consonants and natural inter-syllable closures inside speech.
    # Only extended pauses transition to the presenter's same-pose listening face.
    rms = []
    for i in range(count):
        a = max(0, round((i / fps - 0.025) * 16000))
        b = min(len(signal), round((i / fps + 0.025) * 16000))
        chunk = signal[a:b]
        rms.append(float(np.sqrt(np.mean(chunk**2))) if len(chunk) else 0.0)
    level = np.asarray(rms)
    active = level > max(0.0015, min(0.012, np.percentile(level, 95) * 0.04))
    if not active.any():
        return np.zeros(count, np.float32)
    distance = (
        np.min(abs(np.arange(count)[:, None] - np.flatnonzero(active)), axis=1) / fps
    )
    return np.clip((0.15 - distance) / 0.07, 0, 1).astype(np.float32)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("model", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("start", type=float)
    ap.add_argument("count", type=int)
    opt = ap.parse_args()
    np.save(opt.out, extract(opt.source, opt.model, opt.start, opt.count))
