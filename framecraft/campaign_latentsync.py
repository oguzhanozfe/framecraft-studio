"""Optional audio-driven alternative for selectively repairing a visible speaker's take."""

import argparse
import contextlib
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import types
import wave
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

_BACKEND_LOCK = threading.RLock()


def frame_count(seconds, fps):
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Talking take must have a positive finite duration")
    return math.ceil(seconds * fps - 1e-7)


def register(video, model, output):
    """Register the eyebrows, nose, and full outer lips independently of the generator."""
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions

    config = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )
    capture = cv2.VideoCapture(str(video))
    anchors, lips, gap = [], [], []
    left = [70, 63, 105, 66, 107, 46, 53, 52, 65, 55]
    right = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
    outline = [
        61,
        146,
        91,
        181,
        84,
        17,
        314,
        405,
        321,
        375,
        291,
        409,
        270,
        269,
        267,
        0,
        37,
        39,
        40,
        185,
    ]
    try:
        with vision.FaceLandmarker.create_from_options(config) as worker:
            if abs(capture.get(cv2.CAP_PROP_FPS) - 25) > 0.01:
                raise ValueError("Landmark registration requires a 25 fps source")
            while True:
                ok, image = capture.read()
                if not ok:
                    break
                h, w = image.shape[:2]
                rgb = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                )
                detection = worker.detect_for_video(rgb, 40 * len(gap))
                if not detection.face_landmarks:
                    raise ValueError(f"No visible speaker at frame {len(gap)}")
                points = np.array(
                    [(p.x * w, p.y * h) for p in detection.face_landmarks[0]]
                )
                brows = sorted(
                    [points[left].mean(0), points[right].mean(0)], key=lambda p: p[0]
                )
                anchors.append([*brows, points[[1, 4, 98, 327]].mean(0)])
                lips.append(points[outline])
                gap.append(
                    np.linalg.norm(points[13] - points[14])
                    / max(1, np.linalg.norm(points[61] - points[291]))
                )
    finally:
        capture.release()
    if not gap:
        raise ValueError("Empty talking take")
    np.savez(output, anchors=anchors, lips=lips, gap=gap, w=w, h=h)


def calibration(anchors, lips, target, size, mask):
    """Use one translation for the whole take, leaving the actor's articulation intact."""
    sx, sy = size
    aligned = []
    for points, lip in zip(anchors, lips):
        a, b = points - points.mean(0), target - target.mean(0)
        u, _, vt = np.linalg.svd(a.T @ b)
        rotation = vt.T @ u.T
        if np.linalg.det(rotation) < 0:
            vt[-1] *= -1
            rotation = vt.T @ u.T
        linear = b.std() / a.std() * rotation
        aligned.append((lip - points.mean(0)) @ linear.T + target.mean(0))
    aligned = np.asarray(aligned)
    shift = np.array([0.5 * sx, 0.62 * sy]) - np.median(aligned.mean(1), axis=0)
    h, w = mask.shape
    pixels = np.rint((aligned + shift) * np.array([w / sx, h / sy])).astype(int)
    if np.any((pixels < 0) | (pixels >= np.array([w, h]))):
        raise ValueError("Lips extend outside the registered crop")
    covered = float(np.mean(mask[pixels[:, :, 1], pixels[:, :, 0]] < 0.02))
    if covered < 0.99:
        raise ValueError("Full lip outline must be in the model's editable region")
    return shift, covered


def reference_indices(gap, active, source_count):
    """Use disjoint identity footage for speech and a real closed-mouth reference for rests."""
    if source_count < 2 or len(active) > source_count or len(gap) != source_count:
        raise ValueError("Source does not cover the requested audio")
    indices = (np.arange(len(active)) + source_count // 2) % source_count
    closed = np.flatnonzero(gap <= min(0.04, float(np.quantile(gap, 0.12)) + 0.008))
    if np.any(~active) and not len(closed):
        raise ValueError("No closed-mouth reference in this source for silent rests")
    for i in np.flatnonzero(~active):
        indices[i] = closed[np.argmin(np.abs(closed - i))]
    return indices


def speaking_frames(bed, count):
    with wave.open(str(bed), "rb") as stream:
        if stream.getsampwidth() != 2 or stream.getnchannels() != 1:
            raise ValueError("Expected mono PCM16 speech")
        rate = stream.getframerate()
        samples = (
            np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").astype(
                float
            )
            / 32768
        )
    power = np.array(
        [
            np.mean(
                samples[
                    round(i * rate / 25) : max(
                        round(i * rate / 25) + 1, round((i + 1) * rate / 25)
                    )
                ]
                ** 2
            )
            for i in range(count)
        ]
    )
    voiced = np.nan_to_num(power) > 0.008**2
    if not voiced.any():
        raise ValueError("Silent voice bed")
    # Include brief releases and anticipatory closure near speech.
    return (
        np.convolve(voiced.astype(int), np.ones(5, dtype=int), mode="full")[
            2 : 2 + count
        ]
        > 0
    )


def rest_seam(speaking, listening):
    """Find a matching closed expression in both tracked versions of the same take."""
    if speaking.shape != listening.shape or speaking.ndim != 3 or not len(speaking):
        raise ValueError("Same-time presenter tracks required")
    scale = np.maximum(1, np.linalg.norm(listening[:, 33] - listening[:, 263], axis=1))
    opening = (
        np.maximum(
            np.linalg.norm(speaking[:, 13] - speaking[:, 14], axis=1),
            np.linalg.norm(listening[:, 13] - listening[:, 14], axis=1),
        )
        / scale
    )
    mouth = [61, 291, 0, 17, 13, 14]
    upper = [33, 263, 168, 1, 98, 327]
    mouth_error = (
        np.linalg.norm(speaking[:, mouth] - listening[:, mouth], axis=2).mean(1) / scale
    )
    head_error = (
        np.linalg.norm(speaking[:, upper] - listening[:, upper], axis=2).mean(1) / scale
    )
    good = np.flatnonzero(
        (opening < 0.055) & (mouth_error < 0.075) & (head_error < 0.037)
    )
    if not len(good):
        raise ValueError(
            "No genuinely closed, pose-matched frame for the listening transition"
        )
    i = int(good[np.argmin((opening + mouth_error + 2 * head_error)[good])])
    return i, {
        "opening": float(opening[i]),
        "mouth_error": float(mouth_error[i]),
        "head_error": float(head_error[i]),
    }


def join_listening(speech, listening, bed, output, seconds):
    """Join a diffusion utterance to a same-source registered silent tail at a measured closed pose."""
    from .campaign_face import track

    cap = cv2.VideoCapture(str(speech))
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or abs(cap.get(cv2.CAP_PROP_FPS) - 30) > 0.001:
            raise ValueError("Expected speech video at exactly 30 fps")
    finally:
        cap.release()
    total = frame_count(seconds, 30)
    if count >= total:
        raise ValueError("The listening tail must extend the speaking take")
    active = speaking_frames(bed, frame_count(seconds, 25))
    start = math.ceil(((np.flatnonzero(active)[-1] + 1) / 25 + 0.08) * 30)
    window = count - start
    if window < 3:
        raise ValueError("Too few quiet frames to verify the transition")
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        rest = Path(tmp) / "rest.mp4"
        common = ["ffmpeg", "-y", "-nostdin", "-v", "error"]
        encode = [
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "15",
            "-preset",
            "fast",
            "-threads",
            "3",
            "-pix_fmt",
            "yuv420p",
        ]
        subprocess.run(
            [
                *common,
                "-i",
                str(listening),
                "-vf",
                "fps=30",
                "-frames:v",
                str(total),
                *encode,
                str(rest),
            ],
            check=True,
        )

        def geometry(path):
            cap = cv2.VideoCapture(str(path))
            try:
                return tuple(
                    int(cap.get(i))
                    for i in (
                        cv2.CAP_PROP_FRAME_COUNT,
                        cv2.CAP_PROP_FRAME_WIDTH,
                        cv2.CAP_PROP_FRAME_HEIGHT,
                    )
                )
            finally:
                cap.release()

        if geometry(rest) != (total, *geometry(speech)[1:]):
            raise ValueError(
                "Listening footage must match the full delivery interval and geometry"
            )
        a = track(speech, start / 30, window)
        b = track(rest, start / 30, window)
        offset, report = rest_seam(a, b)
        cut = start + offset
        filters = f"[0:v]trim=end_frame={cut},setpts=PTS-STARTPTS[a];[1:v]trim=start_frame={cut}:end_frame={total},setpts=PTS-STARTPTS[b];[a][b]concat=n=2:v=1:a=0"
        target = Path(tmp) / "result.mp4"
        subprocess.run(
            [
                *common,
                "-i",
                str(speech),
                "-i",
                str(rest),
                "-filter_complex",
                filters,
                *encode,
                str(target),
            ],
            check=True,
        )
        if geometry(target) != geometry(rest):
            raise RuntimeError("Could not preserve the entire listening shot")
        os.replace(target, output)
    return dict(**report, cut_frame=cut, frames=total)


class LatentTalk:
    """Load once inside a dedicated GPU worker. Inputs are registered, already cut 25 fps shots."""

    def __init__(self, repo, checkpoints, vae):
        import torch
        from DeepCache import DeepCacheSDHelper
        from diffusers import AutoencoderKL, DDIMScheduler
        from omegaconf import OmegaConf

        self.repo = Path(repo).resolve()
        sys.path.insert(0, str(self.repo))
        # The upstream image processor imports its optional detector at module load.
        # Our processor exclusively uses the measured MediaPipe anchors.
        key = "latentsync.utils.face_detector"
        shim = types.ModuleType(key)
        shim.FaceDetector = None
        with _BACKEND_LOCK:
            previous = sys.modules.get(key)
            sys.modules[key] = shim
            try:
                import latentsync.pipelines.lipsync_pipeline as backend
            finally:
                if previous is None:
                    del sys.modules[key]
                else:
                    sys.modules[key] = previous
        from latentsync.models.unet import UNet3DConditionModel
        from latentsync.whisper.audio2feature import Audio2Feature

        self.backend = backend
        config = OmegaConf.load(self.repo / "configs/unet/stage2_512.yaml")
        self.config = config
        self.audio = Audio2Feature(
            model_path=str(Path(checkpoints) / "whisper/tiny.pt"),
            device="cuda",
            num_frames=config.data.num_frames,
            audio_feat_length=config.data.audio_feat_length,
        )
        image_model = AutoencoderKL.from_pretrained(str(vae), torch_dtype=torch.float16)
        image_model.config.scaling_factor = 0.18215
        image_model.config.shift_factor = 0
        unet, _ = UNet3DConditionModel.from_pretrained(
            OmegaConf.to_container(config.model),
            str(Path(checkpoints) / "latentsync_unet.pt"),
            device="cpu",
        )
        scheduler = DDIMScheduler.from_pretrained(str(self.repo / "configs"))
        self.pipe = backend.LipsyncPipeline(
            vae=image_model,
            audio_encoder=self.audio,
            unet=unet.to(dtype=torch.float16),
            scheduler=scheduler,
        ).to("cuda")
        helper = DeepCacheSDHelper(pipe=self.pipe)
        helper.set_params(cache_interval=3, cache_branch_id=0)
        self.cache = helper

    def render(
        self,
        video,
        bed,
        output,
        landmarks,
        *,
        seconds,
        steps=20,
        guidance=3.0,
        seed=1247,
    ):
        import torch
        from accelerate.utils import set_seed
        from latentsync.utils.affine_transform import AlignRestore
        from latentsync.utils.image_processor import ImageProcessor
        from latentsync.utils.util import read_video_cv2

        from .campaign_cast import digest

        count = frame_count(seconds, 25)
        delivery_frames = frame_count(seconds, 30)
        if abs(seconds - delivery_frames / 30) > 1e-6:
            raise ValueError("Final duration must be representable at 30 fps")
        with wave.open(str(bed), "rb") as stream:
            if abs(stream.getnframes() / stream.getframerate() - seconds) > 0.001:
                raise ValueError("Speech bed must exactly span the requested take")
        with np.load(landmarks) as registered:
            anchors = registered["anchors"]
            lips = registered["lips"]
            gaps = registered["gap"]
            geometry = (int(registered["h"]), int(registered["w"]))
        active = speaking_frames(bed, count)
        donors = reference_indices(gaps, active, len(anchors))
        result = {}

        class RegisteredImage(ImageProcessor):
            def __init__(self, resolution, device, mask_image):
                super().__init__(resolution, device="cpu", mask_image=mask_image)
                self.restorer = AlignRestore(resolution=resolution, device=device)
                shift, covered = calibration(
                    anchors,
                    lips,
                    self.restorer.face_template,
                    self.restorer.face_size,
                    mask_image[0].detach().cpu().numpy(),
                )
                self.restorer.face_template += shift
                self.i = 0
                result.update(mask_coverage=covered, shift_px=shift.tolist())

            def affine_transform(self, image):
                if image.shape[:2] != geometry or self.i >= len(anchors):
                    raise ValueError(
                        "Landmarks no longer match source dimensions or duration"
                    )
                points = anchors[self.i]
                self.i += 1
                face, matrix = self.restorer.align_warp_face(
                    image.copy(), np.round(points), smooth=True
                )
                box = [0, 0, face.shape[1], face.shape[0]]
                face = cv2.resize(
                    face,
                    (self.resolution, self.resolution),
                    interpolation=cv2.INTER_LANCZOS4,
                )
                return torch.from_numpy(face).permute(2, 0, 1), box, matrix

        bank = None
        cursor = 0

        def aligned_source(chunks, frames):
            nonlocal bank
            if (
                len(frames) != len(anchors)
                or len(frames) < count
                or len(chunks) < count
            ):
                raise ValueError(
                    "Insufficient synchronized source, audio features or landmarks"
                )
            del chunks[count:]
            bank, boxes, matrices = self.pipe.affine_transform_video(frames)
            return frames[:count], bank[:count], boxes[:count], matrices[:count]

        encode_ref = self.pipe.prepare_image_latents

        def identity_conditions(images, *args):
            nonlocal cursor
            selected = donors[cursor : cursor + len(images)]
            cursor += len(images)
            return encode_ref(bank[selected.tolist()].float() / 127.5 - 1, *args)

        def audio_tensor(path, *args, **kwargs):
            raw = subprocess.check_output(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    "-i",
                    str(path),
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "-f",
                    "f32le",
                    "-",
                ]
            )
            return torch.from_numpy(np.frombuffer(raw, "<f4").copy())

        output = Path(output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        # Upstream interpolates its final mux paths into a shell command. Use only
        # generated relative ASCII names there; user-supplied paths go to argument arrays.
        with tempfile.TemporaryDirectory(prefix="latentsync-", dir=Path.cwd()) as tmp:
            local = Path(tmp).relative_to(Path.cwd())
            movie = local / "result.mp4"
            with _BACKEND_LOCK, contextlib.ExitStack() as patches:
                set_seed(seed)
                self.cache.enable()
                patches.callback(self.cache.disable)
                for obj, attr, impl in (
                    (self.backend, "ImageProcessor", RegisteredImage),
                    (self.backend, "read_video", lambda p, *a, **k: read_video_cv2(p)),
                    (self.backend, "read_audio", audio_tensor),
                    (self.pipe, "loop_video", aligned_source),
                    (self.pipe, "prepare_image_latents", identity_conditions),
                ):
                    patches.enter_context(patch.object(obj, attr, impl))
                self.pipe(
                    video_path=str(Path(video).resolve()),
                    audio_path=str(Path(bed).resolve()),
                    video_out_path=str(movie),
                    num_frames=self.config.data.num_frames,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    weight_dtype=torch.float16,
                    height=512,
                    width=512,
                    mask_image_path=str(self.repo / "latentsync/utils/mask.png"),
                    temp_dir=str(local / "scratch"),
                )
            if cursor != count or not movie.exists():
                raise RuntimeError("Incomplete diffusion output")
            with tempfile.TemporaryDirectory(dir=output.parent) as encoded:
                target = Path(encoded) / "take.mp4"
                subprocess.run(
                    [
                        "ffmpeg",
                        "-nostdin",
                        "-v",
                        "error",
                        "-y",
                        "-i",
                        str(movie),
                        "-an",
                        "-vf",
                        "fps=30",
                        "-frames:v",
                        str(delivery_frames),
                        "-c:v",
                        "libx264",
                        "-preset",
                        "veryfast",
                        "-crf",
                        "15",
                        "-pix_fmt",
                        "yuv420p",
                        str(target),
                    ],
                    check=True,
                )
                checked = cv2.VideoCapture(str(target))
                try:
                    if int(checked.get(cv2.CAP_PROP_FRAME_COUNT)) != delivery_frames:
                        raise ValueError(
                            "Encoded duration differs from the requested take"
                        )
                finally:
                    checked.release()
                os.replace(target, output)
        return {
            **result,
            "model": "LatentSync 1.6",
            "pipeline": "independent-identity-rest-v1",
            "steps": steps,
            "guidance": guidance,
            "seed": seed,
            "frames": delivery_frames,
            "rest_frames": int((~active).sum()),
            "video_sha256": digest(output),
        }


def render_registered_take(
    source,
    voice,
    output,
    *,
    source_start,
    seconds,
    voice_onset,
    repo,
    checkpoints,
    vae,
    model_python=sys.executable,
    work_dir=None,
    gpu_lock=None,
    seed=1247,
    guidance=3.0,
    steps=20,
):
    """Render an unmodified source interval and speech file using the original audio onset.

    Run from the installed Framecraft environment. The independently installed LatentSync
    environment only handles diffusion. A shared work directory reuses registration for
    different readings of the same take.
    """
    from filelock import FileLock

    from .campaign_cast import digest, voice_bed
    from .campaign_face import landmark_runtime
    from .campaign_listening import render as render_quiet

    source, voice, output = (Path(p).resolve() for p in (source, voice, output))
    if output in (source, voice) or not math.isfinite(source_start) or source_start < 0:
        raise ValueError("Choose a separate result and a valid source start")
    count = frame_count(seconds, 30)
    if (
        abs(seconds - count / 30) > 1e-6
        or not math.isfinite(voice_onset)
        or voice_onset < 0
    ):
        raise ValueError(
            "Choose a frame-accurate duration and nonnegative original voice onset"
        )
    source_hash, voice_hash = digest(source), digest(voice)
    face_python, face_model = landmark_runtime()
    output.parent.mkdir(exist_ok=True, parents=True)
    if work_dir:
        Path(work_dir).mkdir(exist_ok=True, parents=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(
            None, (str(Path(__file__).resolve().parent.parent), env.get("PYTHONPATH"))
        )
    )
    with (
        FileLock(str(output) + ".lock"),
        tempfile.TemporaryDirectory(dir=work_dir) as temp,
    ):
        tmp = Path(temp).resolve()
        identity = json.dumps(
            [source_hash, source_start, seconds, digest(face_model), "mesh-v2"]
        ).encode()
        cache = (Path(work_dir) if work_dir else tmp) / hashlib.sha256(
            identity
        ).hexdigest()[:24]
        cache.mkdir(exist_ok=True)
        base, pose, quiet = (
            cache / p for p in ("picture.mp4", "mesh.npz", "quiet.mp4")
        )
        bed = tmp / "bed.wav"
        voice_bed(voice, bed, seconds, voice_onset)
        last = (
            np.flatnonzero(speaking_frames(bed, frame_count(seconds, 25)))[-1] + 1
        ) / 25
        spoken_seconds = min(float(seconds), math.ceil((last + 0.4) * 30) / 30)
        hybrid = seconds - spoken_seconds >= 0.64
        with FileLock(str(cache / "prepare.lock")):
            if not (pose.is_file() and base.is_file()):
                picture = tmp / "picture.mp4"
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-nostdin",
                        "-v",
                        "error",
                        "-ss",
                        str(source_start),
                        "-i",
                        str(source),
                        "-t",
                        # Extra genuine frames provide an independent identity bank.
                        # Only the first requested interval becomes target footage.
                        str(seconds + 2.0),
                        "-vf",
                        "fps=25",
                        "-an",
                        "-c:v",
                        "libx264",
                        "-crf",
                        "13",
                        "-preset",
                        "fast",
                        "-threads",
                        "3",
                        str(picture),
                    ],
                    check=True,
                )
                registration = tmp / "mesh.npz"
                subprocess.run(
                    [
                        str(face_python),
                        "-m",
                        "framecraft.campaign_latentsync",
                        "register",
                        str(picture),
                        str(face_model),
                        str(registration),
                    ],
                    env=env,
                    check=True,
                )
                os.replace(picture, base)
                os.replace(registration, pose)
            if hybrid:
                proof = quiet.with_suffix(".source.json")
                cached = False
                if quiet.is_file() and proof.is_file():
                    try:
                        receipt = json.loads(proof.read_text(encoding="utf-8"))
                        cached = receipt.get(
                            "source_sha256"
                        ) == source_hash and receipt.get("sha256") == digest(quiet)
                    except (OSError, ValueError):
                        pass
                if not cached:
                    try:
                        render_quiet(source, quiet, start=source_start, seconds=seconds)
                    except ValueError:
                        hybrid = False

        def diffuse(length):
            local_bed = tmp / "segment.wav"
            with (
                wave.open(str(bed), "rb") as reader,
                wave.open(str(local_bed), "wb") as writer,
            ):
                writer.setparams(reader.getparams())
                writer.writeframes(
                    reader.readframes(round(length * reader.getframerate()))
                )
            result = tmp / "spoken.mp4"
            cmd = [
                str(model_python),
                "-m",
                "framecraft.campaign_latentsync",
                "render",
                str(Path(repo).resolve()),
                str(Path(checkpoints).resolve()),
                str(Path(vae).resolve()),
                str(base.resolve()),
                str(local_bed),
                str(pose.resolve()),
                str(result),
                "--seconds",
                str(length),
                "--guidance",
                str(guidance),
                "--seed",
                str(seed),
                "--steps",
                str(steps),
            ]
            lock = FileLock(str(gpu_lock)) if gpu_lock else contextlib.nullcontext()
            with lock:
                proc = subprocess.run(
                    cmd,
                    cwd=tmp,
                    env=env,
                    stdout=subprocess.PIPE,
                    encoding="utf-8",
                    errors="replace",
                    check=True,
                )
            report = json.loads(proc.stdout.strip().splitlines()[-1])
            if report["video_sha256"] != digest(result):
                raise ValueError("Changed or incomplete generated speech")
            return result, report

        result, report = diffuse(spoken_seconds if hybrid else seconds)
        if hybrid:
            try:
                merged = tmp / "joined.mp4"
                seam = join_listening(result, quiet, bed, merged, seconds)
            except ValueError:
                result, report = diffuse(seconds)
            else:
                result = merged
                report.update(
                    pipeline="independent-identity-rest-hybrid-v1",
                    seam=seam,
                    frames=count,
                    speaking_seconds=spoken_seconds,
                    resting_sha256=digest(quiet),
                )
        if digest(source) != source_hash or digest(voice) != voice_hash:
            raise ValueError("Source or voice changed during generation")
        report.update(
            source_sha256=source_hash,
            audio_sha256=voice_hash,
            video_sha256=digest(result),
            source_start=source_start,
            voice_onset=voice_onset,
            seconds=seconds,
        )
        # The output directory can be on a different drive from the model cache.
        with tempfile.TemporaryDirectory(dir=output.parent) as publish:
            import shutil

            staged = Path(publish) / "take.mp4"
            shutil.copyfile(result, staged)
            signature = Path(publish) / "take.json"
            signature.write_text(json.dumps(report, indent=2), encoding="utf-8")
            os.replace(staged, output)
            os.replace(signature, output.with_suffix(".json"))
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    landmarks = sub.add_parser("register")
    for key in ("video", "model", "output"):
        landmarks.add_argument(key, type=Path)
    generate = sub.add_parser("render")
    for key in ("repo", "checkpoints", "vae", "video", "bed", "landmarks", "output"):
        generate.add_argument(key, type=Path)
    generate.add_argument("--seconds", type=float, required=True)
    generate.add_argument("--guidance", type=float, default=3.0)
    generate.add_argument("--steps", type=int, default=20)
    generate.add_argument("--seed", type=int, default=1247)
    take = sub.add_parser(
        "take",
        help="Repair the original moving source with the original voice WAV at its exact onset",
    )
    for key in ("source", "voice", "output"):
        take.add_argument(key, type=Path)
    for key in ("repo", "checkpoints", "vae"):
        take.add_argument("--" + key, type=Path, required=True)
    for key in ("source-start", "seconds", "voice-onset"):
        take.add_argument("--" + key, type=float, required=True)
    take.add_argument("--model-python", type=Path, default=sys.executable)
    take.add_argument("--work-dir", type=Path)
    take.add_argument("--gpu-lock", type=Path)
    take.add_argument("--guidance", type=float, default=3.0)
    take.add_argument("--steps", type=int, default=20)
    take.add_argument("--seed", type=int, default=1247)
    args = parser.parse_args()
    if args.action == "register":
        register(args.video, args.model, args.output)
    elif args.action == "take":
        print(
            json.dumps(
                render_registered_take(
                    **{k: v for k, v in vars(args).items() if k != "action"}
                )
            ),
            flush=True,
        )
    else:
        model = LatentTalk(args.repo, args.checkpoints, args.vae)
        result = model.render(
            args.video,
            args.bed,
            args.output,
            args.landmarks,
            seconds=args.seconds,
            guidance=args.guidance,
            steps=args.steps,
            seed=args.seed,
        )
        print(json.dumps(result), flush=True)
