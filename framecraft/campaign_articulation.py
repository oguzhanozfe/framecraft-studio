"""Measure visible speech on a final camera shot using an independent face mesh.

The audio reference must be the actual aligned dialogue stem for a scored edit,
or recorded camera audio for an unscored shot. Rhythm is not a phoneme transcript.
Optional acoustic anchors use seconds relative to the inspected interval. The tested
CPU model is Google MediaPipe Face Landmarker float16 v1 (face_landmarker.task).
"""

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from .campaign_speech_anchors import acoustic_anchors


def mouth_shape(landmarks, width, height):
    """Inner aperture and lip width in units of the outer eye-corner distance."""
    points = np.array([[p.x * width, p.y * height] for p in landmarks])
    eye_distance = np.linalg.norm(points[33] - points[263])
    if eye_distance < 20:
        return np.nan, np.nan
    # Project inward lips along the normal to the mouth, removing head roll.
    corners = points[291] - points[61]
    mouth_width = np.linalg.norm(corners)
    if mouth_width < 10:
        return np.nan, np.nan
    normal = np.array([-corners[1], corners[0]]) / mouth_width
    gaps = (points[[178, 14, 402]] - points[[81, 13, 311]]) @ normal
    return np.median(np.abs(gaps)) / eye_distance, mouth_width / eye_distance


def face_track(video, landmark_model, start, seconds):
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions

    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(landmark_model)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=2,
    )
    capture = cv2.VideoCapture(str(video))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not capture.isOpened() or not np.isfinite(fps) or fps < 10:
            raise ValueError(f"Cannot measure frame timing in {video}")
        first = round(start * fps)
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        count = total - first if seconds is None else round(seconds * fps)
        # Allow a fractional final frame in the muxed container's duration.
        if (
            first + count == total + 1
            and seconds is not None
            and (start + seconds) * fps - total < 1
        ):
            count = total - first
        if start < 0 or count < 10 or first + count > total:
            raise ValueError("A valid, complete speaking interval is required")
        capture.set(cv2.CAP_PROP_POS_FRAMES, first)
        track = []
        with vision.FaceLandmarker.create_from_options(options) as model:
            for i in range(count):
                ok, bgr = capture.read()
                if not ok:
                    raise ValueError("Could not decode requested speaking interval")
                h, w = bgr.shape[:2]
                scale = min(1.0, 800 / max(h, w))
                if scale < 1:
                    bgr = cv2.resize(
                        bgr,
                        (round(w * scale), round(h * scale)),
                        interpolation=cv2.INTER_AREA,
                    )
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                faces = model.detect_for_video(
                    image, round(1000 * i / fps)
                ).face_landmarks
                track.append(
                    mouth_shape(faces[0], rgb.shape[1], rgb.shape[0])
                    if len(faces) == 1
                    else (np.nan, np.nan)
                )
        return np.array(track), fps
    finally:
        capture.release()


def speech_envelope(audio, audio_start, frames, fps):
    """Vowel-band energy in 40 ms windows centered on each video frame."""
    rate = 16000
    data = subprocess.check_output(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-ss",
            str(audio_start),
            "-i",
            str(audio),
            "-t",
            str(frames / fps),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(rate),
            "-f",
            "f32le",
            "-",
        ]
    )
    samples = np.frombuffer(data, dtype="<f4")
    if audio_start < 0 or len(samples) < rate * (frames - 1) / fps:
        raise ValueError("Dialogue stem ends before the pictured interval")
    # Padding is limited to the analysis window at each end of this actual shot.
    size = 640
    samples = np.pad(samples, (size // 2, size))
    windows = (
        np.arange(size) + np.rint(np.arange(frames) * rate / fps).astype(int)[:, None]
    )
    spectrum = np.abs(np.fft.rfft(samples[windows] * np.hanning(size), axis=1))
    hz = np.fft.rfftfreq(size, 1 / rate)
    return np.sqrt(np.mean(spectrum[:, (hz >= 200) & (hz <= 1500)] ** 2, axis=1))


def _correlation(a, b):
    if min(np.std(a), np.std(b)) < 1e-7:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _round(value):
    return round(float(value), 4)


def _width_contrast(shapes):
    broad = [r["width"] for r in shapes if r["type"] in ("open_vowel", "spread_vowel")]
    rounded = [r["width"] for r in shapes if r["type"] == "rounded_vowel"]
    if len(broad) >= 2 and len(rounded) >= 2:
        return float(np.median(broad) - np.median(rounded))
    return None


def _shape_evidence(anchors, track, good, fps, offset=0.0):
    """Score closures and releases in local time, giving each viseme class equal weight."""
    t = np.arange(len(track)) / fps
    scored = []
    for anchor in anchors:
        kind = anchor["type"]
        at = float(anchor["time"]) + offset
        if kind == "bilabial":
            before, after = (
                (0.075, 0.018) if anchor.get("letter") in ("p", "b") else (0.05, 0.035)
            )
        else:
            before, after = 0.04, 0.045
        nearby = np.flatnonzero(good & (t >= at - before) & (t <= at + after))
        if not len(nearby):
            continue
        lip, wide = track[nearby].T
        if kind == "bilabial":
            gap = float(np.min(lip))
            quality = np.clip(1 - gap / 0.07, 0, 1)
            passed = gap < 0.04
        elif kind == "open_vowel":
            gap = float(np.max(lip))
            target = 0.047 if anchor.get("letter") == "e" else 0.060
            quality = np.clip(gap / (target * 1.35), 0, 1)
            passed = gap >= target
        elif kind in ("rounded_vowel", "spread_vowel"):
            # Tongue position is invisible; use these only as relative shape evidence.
            scored.append(
                {
                    "type": kind,
                    "time": anchor["time"],
                    "letter": anchor.get("letter"),
                    "width": _round(np.median(wide)),
                }
            )
            continue
        else:
            raise ValueError("Unknown acoustic anchor type: " + kind)
        scored.append(
            {
                "type": kind,
                "time": anchor["time"],
                "letter": anchor.get("letter"),
                "gap": _round(gap),
                "width": _round(np.median(wide)),
                "quality": _round(quality),
                "passed": bool(passed),
            }
        )
    groups = {}
    for row in scored:
        if "quality" in row:
            groups.setdefault(row["type"], []).append(row)
    # Every sound gets its own measurement. Weight closure and vowel classes equally.
    overall = (
        float(np.mean([np.mean([r["quality"] for r in rs]) for rs in groups.values()]))
        if groups
        else None
    )
    delta = _width_contrast(scored)
    if overall is not None and delta is not None and "bilabial" not in groups:
        # Contrast inside this same face. An immobile grin scores .5 on this
        # component for all offsets and cannot demonstrate timely lip rounding.
        overall = (overall + float(np.clip(0.5 + delta / 0.08, 0, 1))) / 2
    return scored, groups, overall


def assess_track(track, energy, fps, anchors=()):
    """Assess the same frame-indexed signals in tests and real measurements."""
    track = np.asarray(track, dtype=float)
    energy = np.asarray(energy, dtype=float)
    if track.shape != (len(energy), 2) or fps < 10 or len(energy) < 10:
        raise ValueError("Expected an aperture/width pair and audio value per frame")
    n = len(energy)
    idx = np.arange(n)
    good = np.isfinite(track).all(axis=1)
    base = {
        "version": 1,
        "frames": n,
        "tracked_frames": int(good.sum()),
        "fps": _round(fps),
    }

    def result(status, failures, metrics=None):
        return {
            **base,
            "status": status,
            "passed": status == "pass",
            "failures": failures,
            "metrics": metrics or {},
        }

    if good.mean() < 0.9:
        return result("inconclusive", ["face_coverage"])
    ceiling = np.percentile(energy, 95)
    if ceiling < 1e-4 or not np.isfinite(energy).all():
        return result("inconclusive", ["insufficient_dialogue"])
    amplitude = np.clip(energy / ceiling, 0, 1.5) ** 0.5
    active = amplitude > 0.26
    if active.sum() < max(10, 0.40 * fps):
        return result("inconclusive", ["insufficient_dialogue"])
    if (active & good).sum() < max(8, 0.85 * active.sum()):
        return result("inconclusive", ["face_coverage_during_speech"])
    lip = np.interp(idx, idx[good], track[good, 0])
    width = np.interp(idx, idx[good], track[good, 1])
    # Analyze the cadence within phrases; the long silent tail cannot boost correlation.
    pad = max(1, round(0.085 * fps))
    phrase = np.convolve(active.astype(int), np.ones(2 * pad + 1), mode="same") > 0
    silent = ~phrase & good
    tested = phrase & good
    dynamic = np.percentile(lip[active & good], [10, 90])
    span = dynamic[1] - dynamic[0]
    speed = (
        np.hypot(np.diff(lip, prepend=lip[0]), 0.5 * np.diff(width, prepend=width[0]))
        * fps
    )
    # Match syllable energy to jaw opening; allow normal audiovisual anticipation.
    near = max(1, round(0.075 * fps))
    reach = min(round(0.55 * fps), max(near + 1, n // 4))
    # Use identical voiced observations for competing lags, including near clip edges.
    # If that would remove most of a very short line, narrow only the search radius.
    while (
        reach > near + 1
        and (tested & (idx >= reach) & (idx < n - reach)).sum() < 0.7 * tested.sum()
    ):
        reach -= 1
    indices = idx[tested & (idx >= reach) & (idx < n - reach)]
    if len(indices) < 8:
        return result("inconclusive", ["insufficient_timing_context"])
    measures = {}
    for lag in range(-reach, reach + 1):
        sample = indices[(indices + lag >= 0) & (indices + lag < n)]
        sample = sample[good[sample + lag]]
        if len(sample) >= 8:
            measures[lag] = _correlation(amplitude[sample], lip[sample + lag])
    best_near = max((k for k in measures if abs(k) <= near), key=measures.get)
    best = max(measures, key=measures.get)
    control = [v for k, v in measures.items() if abs(k / fps) >= 0.18]
    contrast = measures[best_near] - np.median(control) if control else 0
    metrics = {
        "voiced_frames": int((active & good).sum()),
        "silent_frames": int(silent.sum()),
        "aperture_p10": _round(dynamic[0]),
        "aperture_p90": _round(dynamic[1]),
        "aperture_range": _round(span),
        "near_speech_correlation": _round(measures[best_near]),
        "control_margin": _round(contrast),
        "best_lag_ms": round(1000 * best / fps),
        "best_correlation": _round(measures[best]),
        "voiced_motion_p75": _round(np.percentile(speed[tested], 75)),
    }
    failures = []
    if span < 0.024:
        failures.append("weak_mouth_articulation")
    weak_rhythm = measures[best_near] < 0.22 or contrast < 0.08
    displaced = abs(best / fps) > 0.15 and measures[best] > measures[best_near] + 0.16
    if displaced:
        # A broad maximum on one loud syllable is not evidence of a shot-wide delay.
        halves = np.array_split(indices, 2)
        differences = [
            _correlation(amplitude[part], lip[part + best])
            - _correlation(amplitude[part], lip[part + best_near])
            for part in halves
        ]
        metrics["split_lag_improvements"] = [_round(v) for v in differences]
        displaced = all(v >= 0.08 for v in differences)
    if silent.sum() >= 0.33 * fps:
        rest_speed = np.percentile(speed[silent], 75)
        metrics["silent_motion_p75"] = _round(rest_speed)
        metrics["silent_aperture_p50"] = _round(np.median(lip[silent]))
        if rest_speed > max(0.32, 1.15 * metrics["voiced_motion_p75"]):
            failures.append("mouth_moves_through_silence")
    separates_controls = False
    if anchors:
        if any(not 0 <= float(row["time"]) < n / fps for row in anchors):
            raise ValueError("Acoustic anchor lies outside this shot")
        shapes, groups, score = _shape_evidence(anchors, track, good, fps)
        if len(shapes) != len(anchors):
            return result("inconclusive", ["missing_anchor_landmarks"], metrics)
        metrics["phonetic_events"] = shapes
        # Rounded vowel widths are meaningful only against broad vowels of the same face.
        delta = _width_contrast(shapes)
        if delta is not None:
            metrics["broad_minus_rounded_width"] = _round(delta)
            if delta < -0.025:
                failures.append("reversed_rounded_vowels")
        if score is not None:
            metrics["viseme_score"] = _round(score)
        for kind, group in groups.items():
            fraction = np.mean([r["passed"] for r in group])
            metrics[kind + "_fraction"] = _round(fraction)
            if fraction < 0.7:
                failures.append(kind + "_mismatch")
        # A stationary or always-open mouth can pass an individual vowel target.
        # Require positive AND negative classes plus mismatched timing controls
        # before allowing phoneme evidence to overcome weak energy correlation.
        contrasting_shapes = bool(groups.get("bilabial")) or (
            delta is not None and delta >= 0.012
        )
        if contrasting_shapes and len(groups.get("open_vowel", ())) >= 2:
            offsets = [-0.48, -0.36, -0.24, 0.24, 0.36, 0.48]
            comparisons = []
            for offset in offsets:
                other, classes, shifted = _shape_evidence(
                    anchors, track, good, fps, offset
                )
                if len(other) == len(shapes) and classes.keys() == groups.keys():
                    comparisons.append(shifted)
            if len(comparisons) >= 3:
                margin = score - np.median(comparisons)
                metrics["viseme_margin_over_shifted"] = _round(margin)
                separates_controls = score >= 0.68 and margin >= 0.13
    if displaced and not separates_controls:
        failures.append("displaced_speech_rhythm")
    if failures:
        return result("fail", failures, metrics)
    if weak_rhythm and not separates_controls:
        return result("inconclusive", ["unmatched_speech_rhythm"], metrics)
    return result("pass", [], metrics)


def measure(
    video,
    audio,
    landmark_model,
    start=0.0,
    seconds=None,
    audio_start=0.0,
    anchors=(),
    *,
    acoustic_model=None,
    transcript=None,
):
    if anchors and acoustic_model is not None:
        raise ValueError("Use one acoustic anchor source")
    if transcript and acoustic_model is None:
        raise ValueError("A transcript requires an acoustic alignment model")
    track, fps = face_track(video, landmark_model, start, seconds)
    energy = speech_envelope(audio, audio_start, len(track), fps)
    if acoustic_model is not None:
        aligned = acoustic_anchors(
            audio,
            acoustic_model,
            start=audio_start,
            seconds=len(track) / fps,
            transcript=transcript,
        )
        anchors = aligned["anchors"]
    result = assess_track(track, energy, fps, anchors)
    if acoustic_model is not None:
        result["acoustic_alignment_mode"] = aligned["mode"]
        result["acoustic_tokens"] = aligned["tokens"]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("video", "audio", "landmark_model"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--audio-start", type=float, default=0)
    parser.add_argument("--seconds", type=float)
    parser.add_argument(
        "--anchors", type=Path, help="JSON list of local acoustic timestamps and types"
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--acoustic-model",
        type=Path,
        help="Local Turkish CTC folder containing config, vocab and model.onnx",
    )
    parser.add_argument(
        "--transcript", help="Actual spoken, expanded line for acoustic alignment"
    )
    args = parser.parse_args()
    if args.anchors and args.acoustic_model:
        parser.error("Use one anchor source per measurement")
    if args.transcript and not args.acoustic_model:
        parser.error("--transcript requires --acoustic-model")
    anchors = (
        json.loads(args.anchors.read_text(encoding="utf8")) if args.anchors else []
    )
    if isinstance(anchors, dict):
        anchors = anchors["anchors"]
    report = measure(
        args.video,
        args.audio,
        args.landmark_model,
        args.start,
        args.seconds,
        args.audio_start,
        anchors,
        acoustic_model=args.acoustic_model,
        transcript=args.transcript,
    )
    encoded = json.dumps(report, indent=2)
    if args.report:
        args.report.write_text(encoded, encoding="utf8")
    print(encoded)
    raise SystemExit(0 if report["passed"] else 2)
