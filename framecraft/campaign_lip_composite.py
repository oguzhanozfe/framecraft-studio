"""Restore a speaker's own lip and facial-hair texture around a synchronized oral cavity."""

import argparse
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from filelock import FileLock

from .campaign_cast import digest
from .campaign_face import track

OUTER = [
    61,
    185,
    40,
    39,
    37,
    0,
    267,
    269,
    270,
    409,
    291,
    375,
    321,
    405,
    314,
    17,
    84,
    181,
    91,
    146,
]
INNER = [
    78,
    191,
    80,
    81,
    82,
    13,
    312,
    311,
    310,
    415,
    308,
    324,
    318,
    402,
    317,
    14,
    87,
    178,
    88,
    95,
]
ENSEMBLE = [33, 133, 362, 263, 168, 6, 197, 1, 98, 327]


def alignment(source, target):
    """Fit an orientation-preserving similarity only to eyes and upper nose."""
    a, b = target[ENSEMBLE].astype(float), source[ENSEMBLE].astype(float)
    ac, bc = a - a.mean(0), b - b.mean(0)
    u, _, vt = np.linalg.svd(ac.T @ bc)
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0:
        raise ValueError("Mismatched presenter pose")
    scale = np.sum((ac @ rot.T) * bc) / max(1, np.sum(ac**2))
    linear = scale * rot
    offset = b.mean(0) - linear @ a.mean(0)
    eye = np.linalg.norm(source[33] - source[263])
    error = np.linalg.norm(a @ linear.T + offset - b, axis=1).mean() / max(eye, 1)
    if not 0.88 < scale < 1.14 or error > 0.045 or eye < 18:
        raise ValueError("The upper faces do not match")
    return np.column_stack([linear, offset]).astype(np.float32), float(eye)


def paint_cell(xmap, ymap, src, dst):
    lower = np.maximum(0, np.floor(dst.min(0)).astype(int))
    upper = np.minimum(xmap.shape[::-1], np.ceil(dst.max(0)).astype(int) + 1)
    x, y = lower
    r, b = upper
    if x >= r or y >= b or abs(cv2.contourArea(dst.astype(np.float32))) < 0.1:
        return
    local = (dst - lower).astype(np.float32)
    m = cv2.getAffineTransform(local, src.astype(np.float32))
    yy, xx = np.mgrid[: b - y, : r - x].astype(np.float32)
    mask = np.zeros((b - y, r - x), np.uint8)
    cv2.fillConvexPoly(mask, np.rint(local * 16).astype(np.int32), 1, shift=4)
    inside = mask.astype(bool)
    xmap[y:b, x:r][inside] = (m[0, 0] * xx + m[0, 1] * yy + m[0, 2])[inside]
    ymap[y:b, x:r][inside] = (m[1, 0] * xx + m[1, 1] * yy + m[1, 2])[inside]


def composite(source, candidate, source_face, candidate_face, lip_shading=0.0):
    """Return the original frame with only anatomically necessary motion and oral pixels."""
    transform, eye = alignment(source_face, candidate_face)
    target_face = candidate_face @ transform[:, :2].T + transform[:, 2]
    source_outer, dest_outer = source_face[OUTER], target_face[OUTER]
    source_inner, dest_inner = source_face[INNER], target_face[INNER]
    if np.linalg.norm(dest_outer - source_outer, axis=1).max() > 0.46 * eye:
        raise ValueError("The lip contours are too far apart")
    # The fixed outer ring contains the union of both lip shapes. Pixels beyond it
    # remain byte-for-byte original. This keeps adjacent stubble, philtrum and cheeks.
    center = (source_outer.mean(0) + dest_outer.mean(0)) / 2
    direction = (source_outer + dest_outer) / 2 - center
    radii = np.linalg.norm(direction, axis=1)
    direction = direction / np.maximum(radii[:, None], 1)
    rad = np.maximum(
        np.sum((source_outer - center) * direction, 1),
        np.sum((dest_outer - center) * direction, 1),
    )
    ring = center + direction * (rad + max(2.0, 0.045 * eye))[:, None]
    left = np.maximum(np.floor(ring.min(0) - 3).astype(int), 0)
    right = np.minimum(np.ceil(ring.max(0) + 4).astype(int), source.shape[1::-1])
    x, y = left
    r, b = right
    if r - x < 5 or b - y < 5:
        raise ValueError("Mouth outside the original frame")
    ym, xm = np.mgrid[y:b, x:r].astype(np.float32)
    # The skin band and the two vermilion strips use this frame of the real actor.
    # Only the inside of the moving mouth comes from the audio-conditioned candidate.
    for s0, s1, d0, d1 in [
        (ring, source_outer, ring, dest_outer),
        (source_outer, source_inner, dest_outer, dest_inner),
    ]:
        for i in range(len(ring)):
            j = (i + 1) % len(ring)
            for source_tri, dest_tri in [
                ([s0[i], s0[j], s1[j]], [d0[i], d0[j], d1[j]]),
                ([s0[i], s1[j], s1[i]], [d0[i], d1[j], d1[i]]),
            ]:
                paint_cell(xm, ym, np.asarray(source_tri), np.asarray(dest_tri) - left)
    texture = cv2.remap(
        source, xm, ym, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    mask = np.zeros((b - y, r - x), np.uint8)
    cv2.fillPoly(
        mask,
        [np.rint((dest_inner - left) * 16).astype(np.int32)],
        255,
        lineType=cv2.LINE_AA,
        shift=4,
    )
    # A closed bilabial has no synthetic teeth. The original lip tissue meets along
    # the registered line. Feather a fraction of a pixel when there is an opening.
    separation = np.linalg.norm(target_face[13] - target_face[14])
    cavity = (mask.astype(np.float32) / 255) * np.clip(
        (separation - 0.003 * eye) / max(0.008 * eye, 0.5), 0, 1
    )
    # Include one anti-aliased contour inside the candidate cavity.
    mat = transform.copy()
    mat[:, 2] -= left
    oral = cv2.warpAffine(
        candidate,
        mat,
        (r - x, b - y),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    if lip_shading:
        # Preserve the mouth render's broad shading when the actor's lipstick or
        # stubble changes the perceived width of a vowel. Only the lip tissue can
        # receive this field. Fine ridges, skin and hair still come from the actor.
        outer = np.zeros_like(mask)
        cv2.fillPoly(
            outer,
            [np.rint((dest_outer - left) * 16).astype(np.int32)],
            255,
            lineType=cv2.LINE_AA,
            shift=4,
        )
        # Saturated or pale cavity lighting must not leak onto closed vermilion.
        # Fade borrowed low-frequency tone in as the actual mouth reopens.
        openness = np.clip((separation / eye - 0.018) / 0.045, 0, 1)
        weight = (outer.astype(np.float32) / 255 * lip_shading * openness)[:, :, None]
        sigma = max(1.2, eye * 0.023)
        shading = cv2.GaussianBlur(
            oral.astype(np.float32), (0, 0), sigma
        ) - cv2.GaussianBlur(texture.astype(np.float32), (0, 0), sigma)
        texture = np.clip(texture.astype(np.float32) + weight * shading, 0, 255)
    alpha = cavity[:, :, None]
    result = source.copy()
    result[y:b, x:r] = np.rint(texture * (1 - alpha) + oral * alpha).astype(np.uint8)
    return result


def render(source, candidate, output, *, source_start, seconds, lip_shading=0.0):
    """Use synchronized picture timestamps and retain exactly the native candidate frame count."""
    if not 0 <= lip_shading <= 1:
        raise ValueError("Lip shading proportion must be between zero and one")
    if (
        not math.isfinite(source_start)
        or not math.isfinite(seconds)
        or source_start < 0
        or seconds <= 0
    ):
        raise ValueError("Use a finite source interval with a positive duration")
    paths = [Path(p).resolve() for p in (source, candidate, output)]
    source, candidate, output = paths
    if len(set(paths)) != 3:
        raise ValueError("Choose separate input and result paths")
    output.parent.mkdir(parents=True, exist_ok=True)
    source_hash, candidate_hash = digest(source), digest(candidate)
    a, b = (cv2.VideoCapture(str(p)) for p in (source, candidate))
    try:
        native, fps = a.get(cv2.CAP_PROP_FPS), b.get(cv2.CAP_PROP_FPS)
        count = int(b.get(cv2.CAP_PROP_FRAME_COUNT))
        if (
            min(native, fps) <= 0
            or count < 1
            or abs(count / fps - seconds) > 1.01 / fps
            or source_start < 0
        ):
            raise ValueError("Timing mismatch")
        w, h = (
            int(b.get(i)) for i in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT)
        )
        if (w, h) != tuple(
            int(a.get(i)) for i in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT)
        ):
            raise ValueError("Speaker picture dimensions changed")
        first = round(source_start * native)
        src_idx = np.floor(np.arange(count) * native / fps + 0.5).astype(int)
        available = int(a.get(cv2.CAP_PROP_FRAME_COUNT)) - first
        if src_idx[-1] >= available:
            raise ValueError("Source does not cover the candidate")
        original_faces = track(source, first / native, int(src_idx[-1] + 1))
        generated_faces = track(candidate, 0, count)
        with (
            FileLock(str(output) + ".lock"),
            tempfile.TemporaryDirectory(dir=output.parent) as tmp,
        ):
            target = Path(tmp) / "take.mp4"
            cmd = [
                "ffmpeg",
                "-nostdin",
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
                "-crf",
                "13",
                "-preset",
                "fast",
                "-threads",
                "3",
                "-pix_fmt",
                "yuv420p",
                str(target),
            ]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            a.set(cv2.CAP_PROP_POS_FRAMES, first)
            prev = -1
            try:
                for i, index in enumerate(src_idx):
                    while prev < index:
                        ok, pic = a.read()
                        if not ok:
                            raise ValueError("Original stream ends early")
                        prev += 1
                    ok, model = b.read()
                    if not ok:
                        raise ValueError("Candidate stream ends early")
                    result = composite(
                        pic,
                        model,
                        original_faces[index],
                        generated_faces[i],
                        lip_shading,
                    )
                    proc.stdin.write(result.tobytes())
                proc.stdin.close()
                if proc.wait():
                    raise RuntimeError("Unable to encode restored lip tissue")
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
                if not proc.stdin.closed:
                    proc.stdin.close()
            if source_hash != digest(source) or candidate_hash != digest(candidate):
                raise ValueError("Input changed during rendering")
            encoded = cv2.VideoCapture(str(target))
            try:
                if (
                    int(encoded.get(cv2.CAP_PROP_FRAME_COUNT)) != count
                    or abs(encoded.get(cv2.CAP_PROP_FPS) - fps) > 0.001
                ):
                    raise ValueError(
                        "Encoded lip repair changed the native frame clock"
                    )
            finally:
                encoded.release()
            proof = {
                "pipeline": "source-vermilion-v1",
                "lip_shading": lip_shading,
                "source_sha256": source_hash,
                "candidate_sha256": candidate_hash,
                "video_sha256": digest(target),
                "source_start": source_start,
                "seconds": seconds,
                "frames": count,
                "fps": fps,
            }
            os.replace(target, output)
            return proof
    finally:
        a.release()
        b.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "candidate", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--source-start", type=float, required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--lip-shading", type=float, default=0.0)
    print(json.dumps(render(**vars(parser.parse_args()))))
