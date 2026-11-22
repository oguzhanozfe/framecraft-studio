"""Correct residual mouth-shape errors in a neural dub using measured speech sounds.

The renderer registers the same actor's real or synthesized mouth poses. Eyes, head
movement, body, and surroundings always come from the current frame.
"""

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import Delaunay

from .campaign_face import POSE, mouth_transform, speech_activity, track

# Outer and inner vermilion border, followed by several surrounding skin points.
BORDER = [
    61,
    40,
    37,
    0,
    267,
    270,
    291,
    321,
    314,
    17,
    84,
    91,
    78,
    81,
    13,
    311,
    308,
    402,
    14,
    178,
]
RING = [2, 98, 97, 123, 147, 172, 152, 397, 376, 352, 326, 327]


def shapes(mesh):
    across = mesh[:, 291] - mesh[:, 61]
    width = np.linalg.norm(across, axis=1)
    eyes = np.maximum(1.0, np.linalg.norm(mesh[:, 33] - mesh[:, 263], axis=1))
    normal = np.stack([-across[:, 1], across[:, 0]], axis=1) / width[:, None]
    difference = mesh[:, [178, 14, 402]] - mesh[:, [81, 13, 311]]
    gap = np.median(np.abs((difference * normal[:, None]).sum(axis=2)), axis=1)
    return np.stack([gap / eyes, width / eyes], axis=1)


def directions(track_shapes, tokens, samples, fps, phonetic_cues=()):
    """Continuous jaw and lip-width targets from the *audio*, including releases."""
    n = len(track_shapes)
    times = np.arange(n) / fps
    activity = speech_activity(samples, n, fps)
    active = activity > 0.65
    if active.sum() < 6:
        raise ValueError("Too little speech to direct a mouth performance")
    volume = np.array(
        [
            np.sqrt(
                np.mean(
                    samples[
                        max(0, round((t - 0.022) * 16000)) : max(
                            1, round((t + 0.025) * 16000)
                        )
                    ]
                    ** 2
                )
            )
            for t in times
        ]
    )
    scale = np.percentile(volume[active], 85)
    intensity = np.clip(volume / max(scale, 0.001), 0, 1)
    resting_width = float(np.median(track_shapes[:, 1]))
    wide = max(np.percentile(track_shapes[active, 1], 80), resting_width * 1.04)
    narrow = min(np.percentile(track_shapes[active, 1], 20), resting_width * 0.87)
    open_gap = float(np.clip(np.percentile(track_shapes[active, 0], 90), 0.125, 0.19))
    # Weak sounds should still permit the lips to form a consonant. A loud
    # nasal or a fricative should never by itself force the jaw open.
    gap = 0.016 + open_gap * (0.12 + 0.62 * intensity)
    width = track_shapes[:, 1].copy()
    for cue in phonetic_cues:
        index = (times >= cue["start"]) & (times < cue["end"])
        if cue["value"] in ("E", "F"):
            width[index] = 0.65 * width[index] + 0.35 * narrow
        if cue["value"] == "X":
            gap[index] *= 0.35
    anchors = []
    for token in tokens:
        letter = token["letter"]
        if token["confidence"] < 0.45 or letter == "|":
            continue
        t = token["time"]
        if letter in "pbm":
            anchors.extend(
                [
                    (t - 0.045, 0.003, resting_width, 0.98),
                    (t - 0.006, 0.003, resting_width, 0.98),
                ]
            )
        elif letter in "aâeöoüuiıî":
            j = int(np.clip(round(t * fps), 0, n - 1))
            if letter in "aâ":
                value, w = open_gap * (0.84 + 0.23 * intensity[j]), wide
            elif letter == "e":
                value, w = open_gap * (0.60 + 0.26 * intensity[j]), wide
            elif letter in "öo":
                value, w = open_gap * (0.58 + 0.20 * intensity[j]), narrow
            elif letter in "üu":
                value, w = open_gap * (0.27 + 0.19 * intensity[j]), narrow * 0.97
            else:
                value, w = open_gap * (0.23 + 0.22 * intensity[j]), wide
            anchors.append((t, value, w, 0.93))
        elif letter in "fv":
            anchors.append((t, 0.019, resting_width, 0.72))
        else:
            anchors.append(
                (
                    t,
                    0.025 + 0.025 * intensity[min(n - 1, round(t * fps))],
                    resting_width,
                    0.36,
                )
            )
    if not anchors:
        raise ValueError("No reliable spoken phonemes in this interval")
    anchors.sort(key=lambda item: item[0])
    # Adjacent nuclei coarticulate. Silence transitions into true listening.
    marks, values, widths, strength = np.asarray(anchors).T
    strength = np.interp(times, marks, strength)
    proximity = np.min(abs(times[:, None] - marks), axis=1)
    strength *= np.clip((0.19 - proximity) / 0.10, 0, 1)
    gap = gap * (1 - strength) + np.interp(times, marks, values) * strength
    width = width * (1 - strength) + np.interp(times, marks, widths) * strength
    gap = gaussian_filter1d(gap, 0.35)
    return np.stack([gap * activity, width], axis=1), activity


def pose_sequence(mesh, geometry, target, activity):
    pose = mesh[:, POSE]
    pose = (pose - pose.mean(axis=1, keepdims=True)) / np.maximum(
        np.linalg.norm(mesh[:, 33] - mesh[:, 263], axis=1), 1
    )[:, None, None]
    n = len(mesh)
    indices = np.arange(n)
    step = np.minimum(abs(indices[:, None] + 1 - indices), 9) * 0.12
    prev = np.zeros(n)
    parents = np.zeros((n, n), np.int32)
    for i in range(n):
        delta = (geometry - target[i]) / [0.026, 0.07]
        cost = np.minimum(np.abs(delta), 8) @ [1.0, 0.65]
        cost += 9 * np.sqrt(
            ((pose - pose[i]) ** 2).mean(axis=(1, 2))
        ) + 0.010 * np.minimum(abs(i - indices), 45)
        if activity[i] < 0.01:
            cost = np.where(indices == i, 0.0, 50.0)
        transition = prev[:, None] + step
        parents[i] = np.argmin(transition, axis=0)
        prev = cost + transition[parents[i], indices]
    path = np.empty(n, np.int32)
    path[-1] = np.argmin(prev)
    for i in range(n - 1, 0, -1):
        path[i - 1] = parents[i, path[i]]
    return path


def register(source, source_points, output_points):
    """Inverse piecewise-affine texture map within the lip and surrounding skin."""
    h, w = source.shape[:2]
    corners = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
    src = np.vstack([source_points, corners])
    dst = np.vstack([output_points, corners])
    tri = Delaunay(dst)
    yy, xx = np.mgrid[:h, :w]
    points = np.column_stack([xx.ravel(), yy.ravel()])
    cell = tri.find_simplex(points)
    transforms = tri.transform[cell]
    a = (transforms[:, :2] @ (points - transforms[:, 2])[:, :, None])[:, :, 0]
    weights = np.column_stack([a, 1 - a.sum(axis=1)])
    coords = (src[tri.simplices[cell]] * weights[:, :, None]).sum(axis=1)
    coords[cell < 0] = points[cell < 0]
    return cv2.remap(
        source,
        coords.reshape(h, w, 2).astype(np.float32),
        None,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def performance_frame(frames, mesh, geometry, target, i, j, activity, strength):
    if activity <= 0.02:
        return frames[i]
    face = mesh[i]
    donor = mesh[j]
    transform = mouth_transform(donor, face)
    if transform is None:
        return frames[i]
    aligned = cv2.transform(donor[None], transform)[0]
    eye = np.linalg.norm(face[33] - face[263])
    center = (face[61] + face[291]) / 2
    h, w = frames[i].shape[:2]
    x, y = np.maximum(0, np.floor(center - [0.53 * eye, 0.34 * eye]).astype(int))
    r, b = np.minimum([w, h], np.ceil(center + [0.53 * eye, 0.36 * eye]).astype(int))
    origin = np.array([x, y], np.float32)
    matrix = transform.copy()
    matrix[:, 2] -= origin
    d = cv2.warpAffine(
        frames[j],
        matrix,
        (r - x, b - y),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    src = frames[i][y:b, x:r]
    a = face[BORDER] - origin
    c = aligned[BORDER] - origin
    err = np.sum(abs((geometry[i] - target) / [0.035, 0.080]) * [1.0, 0.67])
    amount = float(np.clip((err - 0.19) * 0.82, 0, 1) * activity * strength)
    if amount < 0.03:
        return frames[i]
    final = (1 - amount) * a + amount * c
    # Lip rounding changes width even when this short neural take did not
    # supply a fully rounded donor. Keep the adjustment within facial anatomy.
    horizontal = (face[291] - face[61]) / max(1.0, np.linalg.norm(face[291] - face[61]))
    width_ratio = np.clip(target[1] / max(geometry[j, 1], 0.1), 0.85, 1.12)
    center_lip = (final[0] + final[6]) / 2
    final += (
        ((final - center_lip) @ horizontal)[:, None]
        * horizontal
        * (width_ratio - 1)
        * amount
    )
    source_ring = face[RING] - origin
    other_ring = aligned[RING] - origin
    # Local contour registration prevents double edges at teeth and lip borders.
    dst = np.vstack([final, source_ring])
    original = register(src, np.vstack([a, source_ring]), dst)
    replacement = register(d, np.vstack([c, other_ring]), dst)
    texture = np.rint(original * (1 - amount) + replacement * amount).astype(np.uint8)
    mask = np.zeros((b - y, r - x), np.float32)
    hull = cv2.convexHull(np.rint(final[:12]).astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 1)
    margin = max(3, round(0.095 * eye))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1)
    )
    mask = cv2.GaussianBlur(cv2.dilate(mask, kernel), (0, 0), max(1.0, 0.04 * eye))[
        :, :, None
    ]
    out = frames[i].copy()
    out[y:b, x:r] = np.rint(texture * mask + src * (1 - mask)).astype(np.uint8)
    return out


def refine(video, audio, tokens, output, *, phonetic_cues=(), strength=0.94):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    if not frames or not fps:
        raise ValueError("Cannot refine an unreadable speaking performance")
    n = len(frames)
    h, w = frames[0].shape[:2]
    mesh = track(video, 0, n)
    audio_bytes = subprocess.check_output(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(audio),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-f",
            "f32le",
            "-",
        ]
    )
    signal = np.frombuffer(audio_bytes, "<f4")
    if abs(len(signal) / 16000 - n / fps) > 0.12:
        raise ValueError("Mouth performance and voice must cover the same shot")
    signal = np.pad(signal, (0, round(0.08 * 16000)))
    shape = shapes(mesh)
    target, activity = directions(shape, tokens, signal, fps, phonetic_cues)
    chosen = pose_sequence(mesh, shape, target, activity)
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-threads",
        "2",
        "-preset",
        "veryfast",
        "-crf",
        "15",
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for i, j in enumerate(chosen):
            process.stdin.write(
                performance_frame(
                    frames, mesh, shape, target[i], i, j, activity[i], strength
                ).tobytes()
            )
        process.stdin.close()
        if process.wait():
            raise RuntimeError("Could not encode corrected speaking performance")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    return {
        "model": "phoneme-guided neural take",
        "frames": n,
        "fps": fps,
        "strength": strength,
        "token_count": len(tokens),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for key in ("video", "audio", "alignment", "output"):
        ap.add_argument(key, type=Path)
    ap.add_argument("--cues", type=Path)
    ap.add_argument("--strength", type=float, default=0.94)
    args = ap.parse_args()
    tokens = json.loads(args.alignment.read_text(encoding="utf8"))["tokens"]
    cues = (
        json.loads(args.cues.read_text(encoding="utf8"))["mouthCues"]
        if args.cues
        else []
    )
    print(
        refine(
            args.video,
            args.audio,
            tokens,
            args.output,
            phonetic_cues=cues,
            strength=args.strength,
        )
    )
