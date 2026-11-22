"""Retain the speaker's upper-lip hair without substituting the animated mouth."""

from pathlib import Path

import cv2
import numpy as np

from .campaign_face import track
from .campaign_viseme_firm import _encode, _take

UPPER = [61, 40, 37, 0, 267, 270, 291]


def _coordinates(mesh):
    axis = mesh[263] - mesh[33]
    span = float(np.linalg.norm(axis))
    if span < 12:
        raise ValueError("The speaker is too small for upper-lip detail recovery")
    tangent = axis / span
    axes = np.stack([tangent, [-tangent[1], tangent[0]]])
    center = (mesh[61] + mesh[291]) / 2
    local = (mesh - center) @ axes.T / span
    return center, axes, span, local


def _smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def preserve_detail(base, face, original, unedited, *, measured_weight=None):
    """Map the original skin detail to its anatomical strip between nose and lip."""
    middle, axes, eye, points = _coordinates(face)
    anchor, raw_axes, raw_eye, raw_points = _coordinates(unedited)
    if (
        abs(axes[0] @ raw_axes[0]) < 0.92
        or np.linalg.norm(points[1] - raw_points[1]) > 0.20
    ):
        return base
    h, w = base.shape[:2]
    x, y = np.maximum(0, np.floor(middle - eye * np.array([0.52, 0.43])).astype(int))
    r, b = np.minimum(
        [w, h], np.ceil(middle + eye * np.array([0.52, 0.14])).astype(int)
    )
    if min(r - x, b - y) < 12:
        return base
    yy, xx = np.mgrid[y:b, x:r]
    local = (np.stack([xx, yy], -1) - middle) @ axes.T / eye
    u, v = local[:, :, 0], local[:, :, 1]

    def edges(jaw, positions):
        upper = jaw[UPPER]
        ix = np.argsort(upper[:, 0])
        lip = np.interp(positions, upper[ix, 0], upper[ix, 1])
        # Stay on pilose skin above the actual red lip rim. Neither mouth
        # aperture nor expression determines whether follicles should exist.
        lip -= 0.010
        nose = np.minimum(jaw[2, 1] - 0.008, lip - 0.070)
        return lip, nose

    lower, top = edges(points, u)
    source_lower, source_top = edges(raw_points, u)
    taper = 0.016 if measured_weight is not None else 0.028
    weight = _smoothstep((lower - v) / taper) * _smoothstep((v - top + 0.037) / 0.052)
    weight *= _smoothstep((0.40 - abs(u)) / 0.080)
    if np.count_nonzero(weight > 0.20) < 30:
        return base
    progress = np.clip((lower - v) / np.maximum(lower - top, 0.01), -0.20, 1.2)
    source_v = source_lower + progress * (source_top - source_lower)
    coordinates = anchor + raw_eye * (np.stack([u, source_v], -1) @ raw_axes)
    raw = cv2.remap(
        original,
        coordinates[:, :, 0].astype(np.float32),
        coordinates[:, :, 1].astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    ).astype(np.float32)
    rendered = base[y:b, x:r].astype(np.float32)
    blur = max(1.25, 0.035 * eye)
    low_raw = cv2.GaussianBlur(raw, (0, 0), blur)
    low_edit = cv2.GaussianBlur(rendered, (0, 0), blur)
    raw_detail = raw - low_raw
    base_detail = rendered - low_edit
    relevant = weight > 0.5
    if relevant.sum() < 12:
        return base
    # If compression or facial motion already retains the source detail,
    # keep that performance. Low-frequency illumination comes from the edit.
    energy = float(np.mean(np.abs(raw_detail[relevant])))
    existing = float(np.mean(np.abs(base_detail[relevant])))
    if energy < 2.0 or (measured_weight is None and existing > energy * 0.91):
        return base
    fade = np.clip((1 - existing / max(1, energy)) / 0.32, 0, 1)
    if measured_weight is not None:
        # The independent encoded result already showed missing source fibres at
        # this time. Offset mild subpixel resampling blur using the source's own
        # high band. Other frames and the moving vermilion are still untouched.
        fade = measured_weight
        raw_detail += 0.55 * (raw - cv2.GaussianBlur(raw, (0, 0), 0.85))
    change = raw_detail - base_detail + np.clip(low_raw - low_edit, -12, 7) * 0.48
    fixed = base.copy()
    fixed[y:b, x:r] = np.clip(
        np.rint(rendered + change * (weight * fade)[:, :, None]), 0, 255
    ).astype(np.uint8)
    return fixed


def restore(source, video, output, *, source_start, failure_times=None):
    """Read each corresponding unmodified camera frame and retain its skin detail."""
    source = Path(source)
    if not np.isfinite(source_start) or source_start < 0:
        raise ValueError("Invalid source start")
    base, mesh, fps = _take(Path(video))
    cap = cv2.VideoCapture(str(source))
    try:
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if source_fps <= 0:
            raise ValueError("The unedited source must be readable")
        first = round(source_start * source_fps)
        ix = np.rint(np.arange(len(base)) / fps * source_fps).astype(int)
        if first + ix[-1] >= int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) + max(
            1, round(0.04 * source_fps)
        ):
            raise ValueError("The unedited source does not cover the selected take")
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        original = []
        for _ in range(ix[-1] + 1):
            ok, frame = cap.read()
            if not ok:
                break
            original.append(frame)
    finally:
        cap.release()
    if not original:
        raise ValueError("Could not decode the required segment of the unedited source")
    raw_mesh = track(source, first / source_fps, len(original))
    ix = np.minimum(ix, len(original) - 1)
    weights = None
    if failure_times is not None:
        marks = np.asarray(failure_times, dtype=float)
        if marks.ndim != 1 or not len(marks) or not np.isfinite(marks).all():
            raise ValueError("Measured upper-lip failure times are required")
        times = np.arange(len(base)) / fps
        distance = np.min(abs(times[:, None] - marks), axis=1)
        weights = _smoothstep(1 - np.maximum(0, distance - 0.035) / 0.055)
    changed = 0

    def frames():
        nonlocal changed
        for i, j in enumerate(ix):
            if weights is not None and weights[i] == 0:
                yield base[i]
                continue
            frame = preserve_detail(
                base[i],
                mesh[i],
                original[j],
                raw_mesh[j],
                measured_weight=None if weights is None else weights[i],
            )
            changed += frame is not base[i]
            yield frame

    h, w = base[0].shape[:2]
    _encode(output, fps, (w, h), frames())
    return {
        "model": "matched original upper-lip skin detail",
        "pipeline": "source-follicle-detail-v1"
        if weights is None
        else "local-source-follicle-detail-v1",
        "frames": len(base),
        "fps": fps,
        "restored_frames": int(changed),
    }
