"""Check a decoded face edit for newly displaced cheek, chin, or hair contours."""

import argparse
import math
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from .campaign_experiments import fingerprint
from .campaign_face import LOWER_FACE, track
from .campaign_lip_texture import _texture_metrics, _texture_verdict
from .core import save

FACE_INTEGRITY_VERSION = 5


def _cheek_metrics(original, rendered, face):
    eye_span = np.linalg.norm(face[33] - face[263])
    if eye_span < 12:
        raise ValueError("The face is too small for a cheek integrity comparison")
    mask = np.zeros(original.shape[:2], np.uint8)
    contour = cv2.convexHull(np.rint(face[LOWER_FACE]).astype(np.int32))
    cv2.fillConvexPoly(mask, contour, 1)
    left, right = face[61], face[291]
    lip_axis = right - left
    # Exclude both lips and a margin for the real movement of the jaw and smile.
    center = tuple(np.rint((left + right) / 2).astype(int))
    axes = (
        round(np.linalg.norm(lip_axis) * 0.52 + eye_span * 0.10),
        round(eye_span * 0.18),
    )
    angle = np.degrees(np.arctan2(lip_axis[1], lip_axis[0]))
    cv2.ellipse(mask, center, axes, angle, 0, 360, 0, -1)
    x, y, w, h = cv2.boundingRect(mask)
    zone = mask[y : y + h, x : x + w].astype(bool)
    a = original[y : y + h, x : x + w]
    b = rendered[y : y + h, x : x + w]
    size = zone.sum()
    if not size:
        raise ValueError("No measurable protected cheek pixels")
    delta = cv2.absdiff(a, b).mean(axis=2)
    changed = (delta > 30) & zone
    _, _, components, _ = cv2.connectedComponentsWithStats(changed.astype(np.uint8), 8)
    edge_a = cv2.Canny(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), 32, 85)
    edge_b = cv2.Canny(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), 32, 85)
    distance = cv2.distanceTransform(255 - edge_a, cv2.DIST_L2, 3)
    new_edges = (edge_b > 0) & (distance > max(1.8, eye_span * 0.035)) & zone
    return {
        "color_error_255": round(float(delta[zone].mean()), 4),
        "changed_fraction": round(float(changed.sum() / size), 5),
        "largest_changed_region": round(
            float(components[1:, 4].max(initial=0) / size), 5
        ),
        "new_contour_fraction": round(float(new_edges.sum() / size), 5),
    }


def _suspect(metrics):
    # A displaced patch adds a new contour AND one connected change outside the lips.
    return (
        metrics["color_error_255"] > 20
        and metrics["changed_fraction"] > 0.2
        and metrics["largest_changed_region"] > 0.12
        and metrics["new_contour_fraction"] > 0.021
    )


class _DecodedFrames:
    """Keep a rolling neighborhood for frame-rate conversion at edit boundaries."""

    def __init__(self, cap, start, size):
        self.cap, self.size = cap, size
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        self.last = start - 1
        self.recent = deque(maxlen=4)

    def get(self, index):
        while self.last < index:
            ok, frame = self.cap.read()
            if not ok:
                raise ValueError("Video ended inside the face comparison")
            self.last += 1
            reduced = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
            self.recent.append((self.last, reduced))
        return next(frame for n, frame in self.recent if n == index)


def assess(source, video, *, start=0.0, video_start=0.0, seconds=None, mesh=None):
    """Compare the actor's unedited take and output at their corresponding shot times."""
    source, video = Path(source), Path(video)
    source_hash, video_hash = fingerprint(source), fingerprint(video)
    src, dst = cv2.VideoCapture(str(source)), cv2.VideoCapture(str(video))
    try:
        fps, vf = src.get(cv2.CAP_PROP_FPS), dst.get(cv2.CAP_PROP_FPS)
        sw, sh = int(src.get(3)), int(src.get(4))
        vw, vh = int(dst.get(3)), int(dst.get(4))
        if min(fps, vf, sw, sh, vw, vh) <= 0:
            raise ValueError("Both face comparison videos must be readable")
        if abs((sw / sh) / (vw / vh) - 1) > 0.025:
            raise ValueError("Face comparison requires the same camera framing")
        if min(start, video_start) < 0 or not all(
            map(math.isfinite, (start, video_start))
        ):
            raise ValueError("Invalid comparison start time")
        first, output_first = round(start * fps), round(video_start * vf)
        available = min(
            (int(src.get(7)) - first) / fps, (int(dst.get(7)) - output_first) / vf
        )
        if seconds is not None and (
            not math.isfinite(seconds)
            or seconds <= 0
            or seconds > available + max(1 / fps, 1 / vf)
        ):
            raise ValueError("Both videos must cover the requested face interval")
        length = min(available, seconds) if seconds is not None else available
        count = min(int(src.get(7)) - first, max(0, round(length * fps)))
        if count < 3:
            raise ValueError("At least three corresponding frames are required")
        if mesh is None:
            mesh = track(source, first / fps, count)
        mesh = np.asarray(mesh)
        if (
            mesh.ndim != 3
            or mesh.shape[1] < 468
            or mesh.shape[2] != 2
            or len(mesh) < count
            or not np.isfinite(mesh[:count]).all()
        ):
            raise ValueError("Incomplete reference face track")
        scale = min(1, 448 / sw, 800 / sh)
        size = (round(sw * scale), round(sh * scale))
        geometry = mesh[:count] * [size[0] / sw, size[1] / sh]
        a = _DecodedFrames(src, first, size)
        b = _DecodedFrames(dst, output_first, size)
        frames = []
        # A 30 fps deliverable can contain a bad frame absent from a 24 fps sampling
        # grid. Decode and assess every output frame, including repeated source poses.
        output_count = min(int(dst.get(7)) - output_first, round(length * vf))
        for index in range(output_count):
            i = min(count - 1, round(index * fps / vf))
            face = geometry[i]
            edited = b.get(output_first + index)
            # H.264 time bases and 24->30 fps conversion may select either neighbor.
            # Use untouched eyes/forehead to select the source without looking at lips.
            center = face[[33, 263]].mean(axis=0)
            span = np.linalg.norm(face[33] - face[263])
            x, y = center
            roi = np.s_[
                max(0, int(y - 0.46 * span)) : min(size[1], int(y + 0.42 * span)),
                max(0, int(x - 0.75 * span)) : min(size[0], int(x + 0.75 * span)),
            ]
            if not edited[roi].size:
                raise ValueError("The tracked face extends outside the picture")
            options = list(range(max(0, i - 1), min(count - 1, i + 1) + 1))
            references = [(index, a.get(first + index)) for index in options]
            ix, reference = min(
                references,
                key=lambda pair: cv2.absdiff(edited[roi], pair[1][roi]).mean(),
            )
            metrics = _cheek_metrics(reference, edited, geometry[ix])
            frames.append(
                {
                    "time": round((output_first + index) / vf, 4),
                    **metrics,
                    "suspect": _suspect(metrics),
                    "lip_texture": _texture_metrics(reference, edited, geometry[ix]),
                }
            )
    finally:
        src.release()
        dst.release()
    flagged = [i for i, f in enumerate(frames) if f["suspect"]]
    # Isolated borderline residuals can be codec noise or a legitimate rapid expression.
    # A sustained defect or a large single-frame flash is visible at playback speed.
    severe = [
        i
        for i in flagged
        if frames[i]["largest_changed_region"] > 0.25
        and frames[i]["new_contour_fraction"] > 0.035
    ]
    recurring = [i for i in flagged if any(j != i and abs(i - j) <= 2 for j in flagged)]
    cheek_times = [frames[i]["time"] for i in sorted(set(severe + recurring))]
    texture = _texture_verdict(frames)
    failure_times = sorted(set(cheek_times + texture["failure_times"]))
    if fingerprint(source) != source_hash or fingerprint(video) != video_hash:
        raise ValueError("The reference or rendered picture changed during comparison")
    return {
        "version": FACE_INTEGRITY_VERSION,
        "passed": not failure_times,
        "frames_checked": len(frames),
        "source_fps": fps,
        "video_fps": vf,
        "source_sha256": source_hash,
        "video_sha256": video_hash,
        "source_start": first / fps,
        "video_start": video_start,
        "failure_times": failure_times,
        "cheek_failure_times": cheek_times,
        "lip_texture": texture,
        "suspect_frames": len(flagged),
        "peak_color_error_255": max(v["color_error_255"] for v in frames),
        "peak_changed_fraction": max(v["changed_fraction"] for v in frames),
        "peak_new_contour_fraction": max(v["new_contour_fraction"] for v in frames),
        "frame_measurements": frames,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("source", "video", "report"):
        parser.add_argument(argument, type=Path)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--video-start", type=float, default=0.0)
    parser.add_argument("--seconds", type=float)
    args = parser.parse_args()
    report = assess(
        args.source,
        args.video,
        start=args.start,
        video_start=args.video_start,
        seconds=args.seconds,
    )
    save(args.report, report)
    raise SystemExit(0 if report["passed"] else 2)
