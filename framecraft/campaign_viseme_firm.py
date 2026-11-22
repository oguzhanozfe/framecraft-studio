"""Repair weak speech shapes with a registered bank of the same presenter's mouth poses.

The head, expressions outside the mouth, framing, timing, and original dialogue
all belong to the incoming shot. Source donor clips must depict the same actor.
"""

import subprocess
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from .campaign_face import POSE, lip_mask, mouth_transform, track
from .campaign_visemes import BORDER, RING, directions, register, shapes


def _take(path):
    capture = cv2.VideoCapture(str(path))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        frames = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    if not frames or fps <= 0:
        raise ValueError("Could not decode a complete presenter take")
    return frames, track(path, 0, len(frames)), fps


def _direct(shape, tokens, samples, fps, cues):
    target, activity = directions(shape, tokens, samples, fps, cues)
    t = np.arange(len(shape)) / fps
    width0 = np.median(shape[:, 1])
    # Use the acoustic phoneme interval as well as its recognition peak.
    # Plosives close on the approach and release as the vowel starts.
    for row in tokens:
        k = row["letter"]
        if row["confidence"] < 0.48 or k not in "aoöuüieâîı":
            continue
        if k in "aâ":
            gap, width = 0.169, width0 * 1.07
        elif k == "e":
            gap, width = 0.123, width0 * 1.065
        elif k in "oö":
            gap, width = 0.103, width0 * 0.865
        elif k in "uü":
            gap, width = 0.052, width0 * 0.83
        else:
            gap, width = 0.043, width0 * 1.05
        first = min(row["start"], row["time"] - 0.022)
        last = max(row["end"], row["time"] + 0.012)
        weight = np.clip(1 - np.maximum(first - t, t - last) / 0.052, 0, 1)
        target = (
            target * (1 - weight[:, None])
            + np.array([gap, width])[None] * weight[:, None]
        )
    for row in tokens:
        k = row["letter"]
        if row["confidence"] < 0.48 or k not in "pbm":
            continue
        first = row["time"] - (0.075 if k in "pb" else 0.05)
        last = row["time"] + (0.003 if k in "pb" else 0.025)
        w = np.clip(1 - np.maximum(first - t, t - last) / 0.027, 0, 1)
        target[:, 0] = target[:, 0] * (1 - w) + 0.0015 * w
    target[:, 0] = gaussian_filter1d(target[:, 0], 0.30)
    return target, activity


def _donors(paths, base_frames, base_mesh):
    frames = []
    mesh = []
    clip = []
    index = []

    def add(sequence, landmarks, ids, j):
        frames.extend(sequence[i] for i in ids)
        mesh.extend(landmarks[i] for i in ids)
        clip.extend([j] * len(ids))
        index.extend(ids)

    add(base_frames, base_mesh, np.arange(len(base_frames)), 0)
    for j, path in enumerate(paths, 1):
        refs, landmarks, _ = _take(path)
        geometry = shapes(landmarks)
        # Sample the duration, keeping actual extrema of both lip closure and rounding.
        ids = np.unique(
            np.r_[
                np.linspace(0, len(refs) - 1, min(42, len(refs)), dtype=int),
                *[np.argsort(geometry[:, k])[np.r_[:5, -5:0]] for k in range(2)],
            ]
        )
        add(refs, landmarks, ids, j)
    return frames, np.asarray(mesh), np.asarray(clip), np.asarray(index)


def _choose(mesh, shapes_, target, activity, clip, index):
    eyes = np.maximum(1, np.linalg.norm(mesh[:, 33] - mesh[:, 263], axis=1))
    pose = mesh[:, POSE]
    pose = (pose - pose.mean(1, keepdims=True)) / eyes[:, None, None]
    # Search across poses of the same face. Keep neighbouring source frames
    # together when they perform the required sound equally well.
    choices = []
    back = []
    past = None
    for i, desired in enumerate(target):
        gp = 0.004 if activity[i] < 0.05 else desired[0]
        wd = desired[1]
        cost = abs(shapes_[:, 0] - gp) / (0.011 if gp < 0.02 else 0.030)
        cost += abs(shapes_[:, 1] - wd) / 0.090
        cost += 11 * np.mean(np.abs(pose - pose[i]), axis=(1, 2)) + 0.09 * (clip != 0)
        shortlist = np.argsort(cost)[:32]
        if past is None:
            value = cost[shortlist]
            parents = np.zeros(len(shortlist), int)
        else:
            old = choices[-1]
            switch = clip[old, None] != clip[None, shortlist]
            distance = np.abs(index[old, None] + 1 - index[None, shortlist])
            trans = past[:, None] + 0.11 * np.minimum(distance, 5) + 0.20 * switch
            parents = trans.argmin(0)
            value = cost[shortlist] + trans[parents, np.arange(len(shortlist))]
        choices.append(shortlist)
        back.append(parents)
        past = value
    at = int(np.argmin(past))
    path = np.zeros(len(target), int)
    for i in range(len(target) - 1, -1, -1):
        path[i] = choices[i][at]
        at = back[i][at]
    return path


def _patch(frames, mesh, geometry, i, j, desired, activity):
    original = frames[i]
    base = mesh[i]
    donor = mesh[j]
    affine = mouth_transform(donor, base)
    if affine is None:
        return original
    mapped = cv2.transform(donor[None], affine)[0]
    eye = np.linalg.norm(base[33] - base[263])
    middle = (base[61] + base[291]) * 0.5
    x, y = np.maximum(0, np.floor(middle - [0.49 * eye, 0.30 * eye]).astype(int))
    h, w = original.shape[:2]
    r, b = np.minimum([w, h], np.ceil(middle + [0.49 * eye, 0.34 * eye]).astype(int))
    if min(r - x, b - y) < 10:
        return original
    origin = np.float32([x, y])
    local = affine.copy()
    local[:, 2] -= origin
    tex = cv2.warpAffine(
        frames[j],
        local,
        (r - x, b - y),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    src = mapped[BORDER] - origin
    outer = src[[0, 6]]
    tangent = outer[1] - outer[0]
    tangent /= max(1, np.linalg.norm(tangent))
    normal = np.array([-tangent[1], tangent[0]], np.float32)
    donor_mid = (src[14] + src[18]) * 0.5
    rest = activity < 0.05
    goal = 0.003 if rest else desired[0]
    # Borrow real teeth, tongue or closed texture, then correct residual aperture.
    # Offset upper/lower borders separately so a collapse never drags the chin.
    delta = np.clip(goal - geometry[j, 0], -0.095, 0.073) * eye
    gap_weights = np.array(
        [
            0,
            -0.12,
            -0.25,
            -0.32,
            -0.25,
            -0.12,
            0,
            0.46,
            0.55,
            0.60,
            0.55,
            0.46,
            0,
            -0.43,
            -0.48,
            -0.43,
            0,
            0.54,
            0.57,
            0.54,
        ],
        np.float32,
    )
    dst = src + gap_weights[:, None] * normal * delta
    ratio = (
        np.clip(desired[1] / max(0.1, geometry[j, 1]), 0.82, 1.14) if not rest else 1
    )
    dst += (((src - donor_mid) @ tangent) * (ratio - 1))[:, None] * tangent
    # Keep global mouth placement from the current head/cheeks, not donor motion.
    dst += ((base[61] + base[291]) / 2 - origin) - outer.mean(0)
    ring = base[RING] - origin
    reg = register(tex, np.vstack([src, mapped[RING] - origin]), np.vstack([dst, ring]))
    roi = original[y:b, x:r]
    mask = np.zeros(roi.shape[:2], np.float32)
    both = np.vstack([dst[:12], base[BORDER[:12]] - origin])
    cv2.fillConvexPoly(mask, cv2.convexHull(np.rint(both).astype(np.int32)), 1)
    radius = max(2, round(0.066 * eye))
    disk = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    mask = cv2.GaussianBlur(cv2.dilate(mask, disk), (0, 0), max(1.0, 0.034 * eye))[
        :, :, None
    ]
    image = original.copy()
    image[y:b, x:r] = np.rint(reg * mask + roi * (1 - mask)).astype(np.uint8)
    return image


def refine(video, audio, tokens, output, *, donors=(), phonetic_cues=()):
    video, output = Path(video), Path(output)
    base, base_mesh, fps = _take(video)
    n = len(base)
    h, w = base[0].shape[:2]
    frames, mesh, clips, indices = _donors(donors, base, base_mesh)
    data = subprocess.check_output(
        [
            "ffmpeg",
            "-nostdin",
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
    signal = np.frombuffer(data, "<f4")
    if abs(len(signal) / 16000 - n / fps) > 0.12:
        raise ValueError("Mouth performance and dialogue durations differ")
    signal = np.pad(signal, (0, 1280))
    geom = shapes(mesh)
    target, activity = _direct(geom[:n], tokens, signal, fps, phonetic_cues)
    sequence = _choose(mesh, geom, target, activity, clips, indices)
    _encode(
        output,
        fps,
        (w, h),
        (
            _patch(frames, mesh, geom, i, j, target[i], activity[i])
            for i, j in enumerate(sequence)
        ),
    )
    return {
        "model": "same-actor phonetic pose bank",
        "pipeline": "phonetic-firm-v1",
        "frames": n,
        "fps": fps,
        "pose_bank_frames": len(mesh),
        "donor_clips": len(donors),
        "measured_targets": sum(t["confidence"] >= 0.48 for t in tokens),
    }


def _encode(output, fps, size, frames):
    w, h = size
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
    encoder = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for frame in frames:
            encoder.stdin.write(frame.tobytes())
        encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError("Could not encode corrected speaking performance")
    finally:
        if encoder.poll() is None:
            encoder.kill()
            encoder.wait()
        if not encoder.stdin.closed:
            encoder.stdin.close()


def _performance_index(time, fps, count, offset):
    """Index the real performance when its visual window matches sound at time+offset."""
    return int(np.clip(np.rint((time - offset) * fps), 0, count - 1))


def _repair_envelope(original, chosen, fps, tokens):
    """A local correction for visibly underplayed closures and stressed nuclei."""
    time = np.arange(len(original)) / fps
    weight = np.zeros(len(original), dtype=float)
    repairs = []
    for row in tokens:
        letter, at = row["letter"], row["time"]
        if row["confidence"] < 0.55 or letter not in "pbmaâe":
            continue
        stop = letter in "pbm"
        # Bilabial stops approach closure before release. Vowel openness peaks
        # across the core of the vowel, frequently after its acoustic onset.
        begin, end = (at - 0.065, at + 0.012) if stop else (at - 0.025, at + 0.043)
        window = (time >= begin - 0.012) & (time <= end + 0.012)
        if not window.any():
            continue
        if stop:
            needs_repair = original[window, 0].min() > 0.023
            improved = chosen[window, 0].min() + 0.009 < original[window, 0].min()
        else:
            goal = 0.064 if letter == "e" else 0.086
            needs_repair = original[window, 0].max() < goal
            improved = chosen[window, 0].max() > original[window, 0].max() + 0.013
        if not needs_repair or not improved:
            continue
        distance = np.maximum(begin - time, time - end)
        fade = np.clip(1 - np.maximum(0, distance) / 0.075, 0, 1)
        weight = np.maximum(weight, fade * fade * (3 - 2 * fade))
        repairs.append({"letter": letter, "time": round(float(at), 4)})
    return weight, repairs


def _head_registration(donor, target):
    d = donor[263] - donor[33]
    t = target[263] - target[33]
    span = float(np.linalg.norm(d))
    other = float(np.linalg.norm(t))
    if min(span, other) < 12 or not 0.78 <= other / span <= 1.28:
        return None
    u = d / span
    v = t / other
    cs = float(u @ v)
    sn = float(u[0] * v[1] - u[1] * v[0])
    if cs < 0.94:
        return None
    # Eye alignment supplies a reliable scale even when a strong rounded
    # vowel makes a candidate's mouth less than half its ordinary width.
    left = (donor[[33, 263]].mean(0) - donor[1]) / span
    right = (target[[33, 263]].mean(0) - target[1]) / other
    if np.linalg.norm(left - right) > 0.16:
        return None
    rot = np.float32([[cs, -sn], [sn, cs]])
    axis = target[291] - target[61]
    width = float(np.linalg.norm(axis))
    native_width = float(np.linalg.norm(donor[291] - donor[61]))
    if min(width, native_width) < 4:
        return None
    unit = axis / width
    # Limit horizontal compression. Collapsing a full lip to a point removes
    # visible upper-lip tissue even when the inner seam closes correctly.
    compression = np.clip(width / (native_width * other / span), 0.83, 1.10)
    linear = (
        (np.eye(2, dtype=np.float32) + (compression - 1) * np.outer(unit, unit))
        @ rot
        * (other / span)
    )
    middle = (target[61] + target[291]) / 2
    return np.column_stack([linear, middle - linear @ ((donor[61] + donor[291]) / 2)])


def _transplant(base, mesh, donor, donor_mesh, weight, *, head_scale=False):
    """Transfer the complete moving lip image between compatible views of one actor."""
    if weight <= 0:
        return base
    eye = np.linalg.norm(mesh[263] - mesh[33])
    center = (mesh[61] + mesh[291]) * 0.5
    if head_scale:
        affine = _head_registration(donor_mesh, mesh)
        if affine is None:
            return base
    else:
        affine = mouth_transform(donor_mesh, mesh)
        if affine is None:
            return base
        # Mouth corners move with the spoken vowel. Estimate scale from the stable
        # eye baseline while retaining the mouth estimate of rotation and centre.
        eye = np.linalg.norm(mesh[263] - mesh[33])
        donor_eye = np.linalg.norm(donor_mesh[263] - donor_mesh[33])
        scale = np.linalg.norm(affine[:, 0])
        linear = affine[:, :2] * np.clip((eye / max(1, donor_eye)) / scale, 0.81, 1.20)
        center = (mesh[61] + mesh[291]) * 0.5
        affine = np.column_stack(
            [linear, center - linear @ ((donor_mesh[61] + donor_mesh[291]) * 0.5)]
        )
    mapped = cv2.transform(donor_mesh[None], affine)[0]
    h, w = base.shape[:2]
    lower = np.maximum(0, np.floor(center - eye * np.array([0.53, 0.35])).astype(int))
    upper = np.minimum(
        [w, h], np.ceil(center + eye * np.array([0.53, 0.36])).astype(int)
    )
    x, y = lower
    r, b = upper
    if min(r - x, b - y) < 10:
        return base
    affine[:, 2] -= lower
    tex = cv2.warpAffine(
        donor,
        affine,
        (r - x, b - y),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    box = (x, y, r, b)
    mask = np.maximum(lip_mask(mesh, box), lip_mask(mapped, box))
    roi = base[y:b, x:r]
    # Correct only illumination mismatch at surrounding skin. Never fit lip colour.
    boundary = (mask[:, :, 0] > 0.06) & (mask[:, :, 0] < 0.45)
    if boundary.sum() >= 16:
        diff = (roi[boundary].astype(np.float32) - tex[boundary]).mean(axis=0)
        tex = np.clip(tex.astype(np.float32) + np.clip(diff, -12, 12), 0, 255)
    mask *= weight
    result = base.copy()
    result[y:b, x:r] = np.rint(tex * mask + roi * (1 - mask)).astype(np.uint8)
    return result


def _rounding_ratios(geometry, tokens, fps):
    """Use Turkish vowel rounding to direct existing neural mouth textures."""
    times = np.arange(len(geometry)) / fps
    rounded = [
        (t["letter"], t["time"])
        for t in tokens
        if t["confidence"] >= 0.55 and t["letter"] in "oöuü"
    ]
    wide = [
        t["time"] for t in tokens if t["confidence"] >= 0.55 and t["letter"] in "aeiâî"
    ]
    if not rounded or not wide:
        return np.ones(len(times))
    wide_width = np.median([geometry[np.argmin(abs(times - t)), 1] for t in wide])
    desired = np.ones(len(times))
    for letter, at in rounded:
        # A high back rounded vowel rounds more tightly than an open-mid vowel.
        target = wide_width * (0.86 if letter in "uü" else 0.90)
        ratio = np.clip(target / np.maximum(geometry[:, 1], 0.15), 0.74, 1.0)
        # Anticipatory coarticulation peaks through the acoustic vowel core.
        distance = np.maximum(at - 0.044 - times, times - at - 0.045)
        weight = np.clip(1 - np.maximum(0, distance) / 0.065, 0, 1)
        desired = np.minimum(desired, 1 - (1 - ratio) * weight**2 * (3 - 2 * weight))
    return desired


def _round_lips(frame, face, ratio):
    if ratio >= 0.994:
        return frame
    across = face[291] - face[61]
    eye = np.linalg.norm(face[263] - face[33])
    tangent = across / max(1, np.linalg.norm(across))
    center = (face[61] + face[291]) / 2
    h, w = frame.shape[:2]
    x, y = np.maximum(0, np.floor(center - [0.53 * eye, 0.34 * eye]).astype(int))
    r, b = np.minimum([w, h], np.ceil(center + [0.53 * eye, 0.35 * eye]).astype(int))
    if min(r - x, b - y) < 10:
        return frame
    origin = np.float32([x, y])
    src = face[BORDER] - origin
    dst = src + (((src - center + origin) @ tangent) * (ratio - 1))[:, None] * tangent
    ring = face[RING] - origin
    roi = frame[y:b, x:r]
    warped = register(roi, np.vstack([src, ring]), np.vstack([dst, ring]))
    mask = lip_mask(face, (x, y, r, b))
    corrected = frame.copy()
    corrected[y:b, x:r] = np.rint(warped * mask + roi * (1 - mask)).astype(np.uint8)
    return corrected


def repair_neural(video, pose_video, tokens, output, *, offset=0.0):
    """Preserve a neural utterance; repair only measured local defects with its same-actor take.

    Both input shots must share a source take and timed voice bed. `offset` is
    the measured signed acoustic-to-visual lag in seconds. No audio is rewritten.
    """
    if not np.isfinite(offset) or abs(offset) > 0.32:
        raise ValueError("Unexpected timing correction for a matched speaking take")
    base, base_mesh, fps = _take(video)
    poses, pose_mesh, pose_fps = _take(pose_video)
    n = len(base)
    if abs(n / fps - len(poses) / pose_fps) > 0.09:
        raise ValueError("Mismatch between source and correction time spans")
    times = np.arange(n) / fps
    native = [_performance_index(t, fps, n, offset) for t in times]
    donor = [_performance_index(t, pose_fps, len(poses), 0) for t in times]
    weight, repairs = _repair_envelope(
        shapes(base_mesh)[native], shapes(pose_mesh)[donor], fps, tokens
    )
    rounding = _rounding_ratios(shapes(base_mesh)[native], tokens, fps)

    def render():
        for i, (j, k) in enumerate(zip(native, donor)):
            pix = base[i]
            if i != j:
                pix = _transplant(pix, base_mesh[i], base[j], base_mesh[j], 1)
            pix = _transplant(pix, base_mesh[i], poses[k], pose_mesh[k], weight[i])
            yield _round_lips(pix, base_mesh[i], rounding[i])

    h, w = base[0].shape[:2]
    _encode(output, fps, (w, h), render())
    return {
        "model": "same-actor neural articulation with local pose corrections",
        "pipeline": "registered-phoneme-repair-v2",
        "rounded_frames": int(np.sum(rounding < 0.994)),
        "fps": fps,
        "frames": n,
        "estimated_speech_offset": offset,
        "local_repairs": repairs,
        "locally_modified_fraction": round(float(np.mean(weight > 0.01)), 4),
    }


def restore_lip_texture(video, donor, failures, tokens, output):
    """Use another performance of the same take and voice only around visible texture defects."""
    base, mesh, fps = _take(video)
    pictures, secondary_mesh, secondary_fps = _take(donor)
    n = len(base)
    if abs(n / fps - len(pictures) / secondary_fps) > 0.09:
        raise ValueError("Mouth texture replacement requires the same timed shot")
    times = np.arange(n) / fps
    marks = np.asarray(failures, dtype=float)
    if not len(marks) or not np.isfinite(marks).all():
        raise ValueError("The defective frames must be explicitly identified")
    distance = np.min(abs(times[:, None] - marks), axis=1)
    fade = np.clip(1 - np.maximum(0, distance - 0.045) / 0.073, 0, 1)
    fade = fade * fade * (3 - 2 * fade)
    geometry = shapes(secondary_mesh)
    primary = shapes(mesh)
    stop = np.asarray(
        [t["time"] for t in tokens if t["confidence"] >= 0.55 and t["letter"] in "mpb"],
        dtype=float,
    )
    donor_times = np.arange(len(pictures)) / secondary_fps

    def render():
        for i, t in enumerate(times):
            if fade[i] < 0.01:
                yield base[i]
                continue
            possible = np.where(np.abs(donor_times - t) < 0.13)[0]
            desired = primary[i, 0]
            if len(stop) and np.any((t >= stop - 0.065) & (t <= stop + 0.025)):
                desired = min(desired, 0.007)
            # Source-coordinate centres already match, so use many consecutive
            # natural frames. Slightly shift the donor when articulation requires it.
            cost = (
                np.abs(geometry[possible, 0] - desired) / 0.035
                + np.abs(donor_times[possible] - t) / 0.065
            )
            j = possible[int(np.argmin(cost))]
            yield _transplant(
                base[i],
                mesh[i],
                pictures[j],
                secondary_mesh[j],
                fade[i],
                head_scale=True,
            )

    h, w = base[0].shape[:2]
    _encode(output, fps, (w, h), render())
    return {
        "model": "same-actor genuine lip-texture replacement",
        "pipeline": "registered-lip-texture-v2",
        "frames": n,
        "fps": fps,
        "restored_frame_fraction": round(float(np.mean(fade > 0.01)), 4),
        "measured_texture_failure_times": marks.tolist(),
    }


def repair_from_report(
    video, donor, tokens, report, output, *, video_receipt, donor_receipt
):
    """Generate a same-shot alternative from an encoded candidate's measured texture defects."""
    from .campaign_experiments import fingerprint

    if report.get("video_start", 0) != 0:
        raise ValueError(
            "Repair the standalone presenter take before rebuilding the complete edit"
        )
    video_hash = fingerprint(Path(video))
    if (
        report.get("video_sha256") != video_hash
        or video_receipt.get("video_sha256") != video_hash
    ):
        raise ValueError("Texture report refers to different video bytes")
    if donor_receipt.get("video_sha256") != fingerprint(Path(donor)):
        raise ValueError("Texture donor changed after it was signed")
    for key in ("source_sha256", "audio_sha256"):
        if not video_receipt.get(key) or video_receipt.get(key) != donor_receipt.get(
            key
        ):
            raise ValueError(
                "Texture repair requires the same camera take and dialogue"
            )
    for key in ("source_start", "voice_onset", "seconds"):
        if abs(float(video_receipt[key]) - float(donor_receipt[key])) > 0.04:
            raise ValueError("Texture donor has different picture or speech timing")
    defects = report["lip_texture"]["defects"]
    times = sorted(
        {
            t
            for kind in ("pale_contraction", "foreign_dark_patch")
            for t in defects.get(kind, ())
        }
    )
    if not times:
        raise ValueError(
            "This candidate has no reported local lip-colour or dark-patch defect"
        )
    result = restore_lip_texture(video, donor, times, tokens, output)
    result.update(
        parent_video=video_hash,
        texture_donor=donor_receipt["video_sha256"],
        texture_gate_version=report["lip_texture"]["version"],
        source_sha256=video_receipt["source_sha256"],
        audio_sha256=video_receipt["audio_sha256"],
    )
    return result
