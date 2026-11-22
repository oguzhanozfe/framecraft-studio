"""Identify the actual spoken line against same-speaker alternative dialogue.

Compare pretrained visual and acoustic embeddings on identical decoded mouth
windows. Alternatives are onset- and level-matched. Reordering the target's
own voiced windows supplies an additional control with identical audio content.
Run this alongside physical articulation checks; this comparison cannot tell
whether a particular lip closure looks anatomically convincing.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .campaign_sync import (
    acoustic_features,
    load_model,
    observed_features,
    read_pcm,
)

RATE = 16000
HOP = 640


def _bounds(pcm):
    """Active 10 ms blocks; keep an absolute floor for inaudible inputs."""
    block = 160
    data = np.pad(pcm.astype(np.float64), (0, (-len(pcm)) % block))
    if not len(data):
        return None
    rms = np.sqrt(np.mean(data.reshape(-1, block) ** 2, axis=1))
    top = np.percentile(rms, 98)
    active = np.flatnonzero(rms > max(75, 0.045 * top))
    if not len(active):
        return None
    return int(active[0] * block), min(len(pcm), int((active[-1] + 1) * block))


def match_onset(target, other):
    original, alternate = _bounds(target), _bounds(other)
    if not original or not alternate:
        return None
    first, end = original
    from_, to = alternate
    ratio = (to - from_) / (end - first)
    if not 0.55 <= ratio <= 1.8:
        return None
    duration = min(to - from_, len(target) - first)
    foil = other[from_ : from_ + duration].astype(float)
    reference = target[first:end].astype(float)
    gain = np.sqrt(np.mean(reference**2) / max(1.0, np.mean(foil**2)))
    bed = np.zeros_like(target)
    bed[first : first + duration] = np.clip(foil * gain, -32768, 32767).astype(
        target.dtype
    )
    return bed


def _voiced_windows(pcm, windows):
    bounds = _bounds(pcm)
    if not bounds:
        return np.zeros(windows, dtype=bool)
    powers = np.array(
        [
            np.mean((pcm[i * HOP : (i + 5) * HOP].astype(float) / 32768) ** 2)
            for i in range(windows)
        ]
    )
    return powers > max(0.000035, 0.015 * np.percentile(powers, 95))


def _best(visual, acoustic, indices, lags, permutation=None):
    # Every hypothesis receives the identical frames for both modalities.
    alternatives = []
    for lag in lags:
        sound = acoustic[indices + lag]
        if permutation is not None:
            sound = sound[permutation]
        distances = np.linalg.norm(visual[indices] - sound, axis=1)
        alternatives.append((float(distances.mean()), lag, distances))
    return min(alternatives, key=lambda item: item[0])


def compare_features(visual, target, other, active):
    """Lower distances correspond to a better match. Non-overlap is inconclusive."""
    n = min(len(visual), len(target), len(other), len(active))
    radius = min(5, (n - 10) // 2)
    if radius < 2:
        return {"status": "inconclusive", "reason": "too_few_comparable_windows"}
    indices = np.flatnonzero(
        active[:n] & (np.arange(n) >= radius) & (np.arange(n) < n - radius)
    )
    if len(indices) < 12 or indices[-1] - indices[0] < 13:
        return {
            "status": "inconclusive",
            "reason": "too_few_comparable_windows",
            "voiced_windows": len(indices),
        }
    real, real_lag, real_windows = _best(visual, target, indices, range(-2, 3))
    foil, foil_lag, foil_windows = _best(
        visual, other, indices, range(-radius, radius + 1)
    )
    sequence = np.arange(len(indices))
    permutations = [
        np.roll(sequence, len(sequence) // 3),
        np.roll(sequence, len(sequence) // 2),
    ]
    # Include a within-line order reversal that keeps short acoustic contexts intact.
    blocks = [sequence[k : k + 4] for k in range(0, len(sequence), 4)]
    permutations.append(np.concatenate(blocks[::-1]))
    deranged = min(
        _best(visual, target, indices, range(-2, 3), order)[0] for order in permutations
    )
    margin = foil - real
    fraction = float(np.mean(real_windows < foil_windows))
    temporal = deranged - real
    if margin < -0.35 and fraction < 0.43:
        status, reason = "fail", "alternative_fits_better"
    elif margin < 0.45 or fraction < 0.58:
        status, reason = "inconclusive", "indistinguishable_alternative"
    elif temporal < 0.35:
        status, reason = "inconclusive", "indistinguishable_syllable_order"
    else:
        status, reason = "pass", None
    return {
        "status": status,
        "reason": reason,
        "voiced_windows": len(indices),
        "target_distance": round(real, 3),
        "target_offset_ms": real_lag * 40,
        "alternative_distance": round(foil, 3),
        "alternative_offset_ms": foil_lag * 40,
        "target_advantage": round(margin, 3),
        "fraction_preferring_target": round(fraction, 3),
        "target_order_advantage": round(temporal, 3),
    }


def identify(
    video,
    audio,
    foils,
    detector_path,
    model_dir,
    *,
    start=0.0,
    seconds=None,
    audio_start=0.0,
):
    if not foils:
        raise ValueError("Provide at least one other spoken line from this speaker")
    visible, target, pcm, detected, count, seconds = observed_features(
        video, audio, detector_path, model_dir, start, seconds, audio_start
    )
    result = {
        "version": 1,
        "method": "same-face same-speaker counterfactual lines and syllable order",
        "passed": False,
        "status": "inconclusive",
        "failures": [],
        "comparisons": [],
        "frames": count,
        "detected": detected,
    }
    if visible is None:
        result["failures"].append("face_coverage")
        return result
    if _bounds(pcm) is None:
        result["failures"].append("inaudible_target")
        return result
    model = load_model(model_dir)
    target = target.detach().cpu().numpy()
    visible = visible.detach().cpu().numpy()
    original_voiced = _voiced_windows(pcm, len(target))
    for index, item in enumerate(foils):
        source = item if isinstance(item, dict) else {"audio": item}
        alt = read_pcm(
            source["audio"], source.get("start", 0), source.get("seconds", seconds)
        )
        other = match_onset(pcm, alt)
        if other is None:
            result["comparisons"].append(
                {
                    "alternative": index,
                    "status": "inconclusive",
                    "reason": "duration_or_activity",
                }
            )
            continue
        features = acoustic_features(model, other, len(target)).detach().cpu().numpy()
        active = original_voiced & _voiced_windows(other, len(target))
        compared = compare_features(visible, target, features, active)
        result["comparisons"].append({"alternative": index, **compared})
    reports = result["comparisons"]
    eligible = [v for v in reports if "target_advantage" in v]
    result["failures"] = [v["reason"] for v in reports if v.get("reason")]
    if any(v["status"] == "fail" for v in eligible):
        result["status"] = "fail"
    elif eligible and all(v["status"] == "pass" for v in eligible):
        result.update(status="pass", passed=True, failures=[])
    if eligible:
        result["target_distance"] = round(
            float(np.mean([v["target_distance"] for v in eligible])), 3
        )
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("video", "audio", "detector", "model_dir"):
        ap.add_argument(name, type=Path)
    ap.add_argument("--foil", type=Path, action="append", required=True)
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--seconds", type=float)
    ap.add_argument("--audio-start", type=float, default=0)
    ap.add_argument("--report", type=Path)
    ap.add_argument("--threads", type=int, default=3, help="CPU inference threads")
    args = ap.parse_args()
    if args.threads < 1:
        ap.error("--threads must be positive")
    import torch

    torch.set_num_threads(args.threads)
    out = identify(
        args.video,
        args.audio,
        args.foil,
        args.detector,
        args.model_dir,
        start=args.start,
        seconds=args.seconds,
        audio_start=args.audio_start,
    )
    encoded = json.dumps(out, indent=2)
    print(encoded)
    if args.report:
        args.report.write_text(encoded, encoding="utf8")
    raise SystemExit(0 if out["passed"] else 2)
