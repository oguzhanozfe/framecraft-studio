"""Render source-grounded campaigns and assess the decoded delivery files."""

import gzip
import hashlib
import html
import io
import json
import math
import os
import secrets
import shutil
import subprocess
import wave
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Literal

import av
import cv2
import numpy as np
import onnxruntime as ort
from filelock import FileLock
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

from .audit import recognizer, similarity
from .campaign import timeline
from .core import read, save
from .director import ask
from .edit import ffmpeg, plate
from .providers import generate
from .quality import audio_metrics, glyph_quality, glyph_template, transcript

W, H, FPS, RATE = 720, 1280, 24, 48000
cv2.setNumThreads(2)


def fingerprint(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def fit(frame, *, creator=False):
    h, w = frame.shape[:2]
    if creator or abs(w / h - W / H) < 0.045:
        factor = max(W / w, H / h)
        img = cv2.resize(frame, (max(W, round(w * factor)), max(H, round(h * factor))))
        y, x = (img.shape[0] - H) // 2, (img.shape[1] - W) // 2
        return img[y : y + H, x : x + W].copy()
    factor = min(W / w, H / h)
    cw, ch = min(W, round(w * factor)), min(H, round(h * factor))
    fg = cv2.resize(
        frame, (cw, ch), interpolation=cv2.INTER_AREA if factor < 1 else cv2.INTER_CUBIC
    )
    # Continue the actual edge colours outside the view; keep the entire reference visible.
    y, x = (H - ch) // 2, (W - cw) // 2
    canvas = cv2.copyMakeBorder(fg, y, H - y - ch, x, W - x - cw, cv2.BORDER_REPLICATE)
    small = cv2.resize(canvas, (45, 80), interpolation=cv2.INTER_AREA)
    canvas = cv2.resize(cv2.GaussianBlur(small, (0, 0), 4), (W, H))
    canvas[y : y + ch, x : x + cw] = fg
    return canvas


@lru_cache(maxsize=1)
def depth_session(model):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 3
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(model), opts, providers=["CPUExecutionProvider"])


def photo_frames(path, box, frames, model, direction):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGBA")
        if box:
            im = im.crop(
                tuple(
                    round(n * size) for n, size in zip(box, (im.width, im.height) * 2)
                )
            )
        elif im.getchannel("A").getextrema()[0] < 255:
            bounds = im.getchannel("A").point(lambda v: 255 if v > 24 else 0).getbbox()
            if bounds:
                im = im.crop(bounds)
        paper = Image.new("RGB", im.size, "white")
        paper.paste(im, mask=im.getchannel("A"))
        base = fit(cv2.cvtColor(np.asarray(paper), cv2.COLOR_RGB2BGR))
    rgb = (
        cv2.cvtColor(cv2.resize(base, (392, 392)), cv2.COLOR_BGR2RGB).astype(np.float32)
        / 255
    )
    rgb = (rgb - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
        [0.229, 0.224, 0.225], np.float32
    )
    session = depth_session(model)
    depth = session.run(
        None, {session.get_inputs()[0].name: rgb.transpose(2, 0, 1)[None]}
    )[0].squeeze()
    depth = cv2.resize(cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX), (W, H))
    yy, xx = np.indices((H, W), dtype=np.float32)
    for i in range(frames):
        t = (1 - math.cos(math.pi * i / (frames - 1))) / 2
        zoom = 1.008 + 0.052 * (t if direction > 0 else 1 - t)
        mx = (xx - W / 2) / zoom + W / 2 + (t - 0.5) * direction * (3 + depth * 5)
        my = (yy - H / 2) / zoom + H / 2 + (t - 0.5) * depth * 4
        yield cv2.remap(
            base, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
        )


def preview_cfr(folder):
    original = folder / "preview.mp4"
    dest = folder / "preview-cfr.mp4"
    if not dest.is_file() or dest.stat().st_mtime_ns < original.stat().st_mtime_ns:
        tmp = folder / "preview-cfr.partial.mp4"
        ffmpeg(
            [
                "-i",
                original,
                "-map",
                "0:v:0",
                "-an",
                "-vf",
                "setpts=PTS-STARTPTS,fps=24",
                "-c:v",
                "libx264",
                "-threads",
                "4",
                "-preset",
                "veryfast",
                "-crf",
                "16",
                tmp,
            ]
        )
        tmp.replace(dest)
    return dest


def moving_frames(
    path, frames, fraction=None, *, creator=False, allow_loop=False, at=None
):
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or fps <= 0 or total <= 0:
            raise ValueError(f"Unreadable footage {path}")
        span = math.floor((frames - 1) * fps / FPS) + 1
        if span > total and not allow_loop:
            raise ValueError(f"{path.name} supplies {total} frames, needs {span}")
        start = round(max(0, total - span) * (fraction or 0))
        if at is not None:
            start = min(max(0, total - span), max(0, round(at * fps)))
        previous = -2
        result = None
        for idx in range(frames):
            pos = (start + int(idx * fps / FPS)) % total
            if pos != previous:
                if pos != previous + 1:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                ok, raw = cap.read()
                if not ok:
                    raise ValueError(f"{path.name} failed decode at {pos}")
                result = fit(raw, creator=creator)
            yield result
            previous = pos
    finally:
        cap.release()


@lru_cache(maxsize=2)
def speaker(path):
    from piper import PiperVoice

    return PiperVoice.load(str(path))


class SpokenLine(BaseModel):
    voice: str = Field(min_length=5, max_length=125)


def narration(folder, item, beats, clock):
    from piper.config import SynthesisConfig

    model = folder.parent / "voices" / f"tr_TR-{item['voice']}-medium.onnx"
    voice = speaker(model)
    pieces = []
    receipts = []
    for idx, (beat, (seconds, _)) in enumerate(zip(beats, clock, strict=True)):
        for attempt in range(3):
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as wav:
                voice.synthesize_wav(
                    beat["voice"], wav, syn_config=SynthesisConfig(length_scale=1.02)
                )
            buffer.seek(0)
            with wave.open(buffer, "rb") as wav:
                rate = wav.getframerate()
                pcm = np.frombuffer(wav.readframes(wav.getnframes()), "<i2")
            active = np.flatnonzero(np.abs(pcm.astype(np.int32)) > 110)
            if not len(active):
                raise ValueError(f"Empty speech in beat {idx + 1}")
            pcm = pcm[max(0, active[0] - rate // 20) : active[-1] + rate // 10]
            natural = len(pcm) / rate
            tempo = max(0.94, natural / (seconds - 0.28))
            if tempo <= 1.28:
                break
            if attempt == 2:
                raise ValueError(
                    f"Speech exceeds beat {idx + 1}: {natural:.1f}s for {seconds}s"
                )
            beat["voice"] = ask(
                SpokenLine,
                "Bu tek Türkçe seslendirme cümlesini aynı anlamla daha kısa söyle. Tek gözleme odaklan. "
                "Başlığa uy. Yeni vaat ekleme. Kısaltmaları ve sayıları konuşma dilinde yaz.",
                {
                    "cümle": beat["voice"],
                    "başlık": beat["caption"],
                    "kanıt": item["proof"],
                    "hedef_kelime": max(4, seconds + 1),
                },
                model_override=item.get("editor_model") or item.get("writer_model"),
            ).voice
        audio = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "s16le",
                "-ar",
                str(rate),
                "-ac",
                "1",
                "-i",
                "pipe:0",
                "-af",
                f"atempo={tempo:.5f},aresample={RATE}",
                "-f",
                "s16le",
                "pipe:1",
            ],
            input=pcm.tobytes(),
            capture_output=True,
            check=True,
        )
        samples = np.frombuffer(audio.stdout, "<i2").astype(np.float32)
        chunk = np.zeros(seconds * RATE, dtype=np.float32)
        offset = int(0.12 * RATE)
        if len(samples) > len(chunk) - offset:
            raise ValueError(f"Speech overflow in beat {idx + 1}")
        chunk[offset : offset + len(samples)] = samples
        pieces.append(chunk)
        receipts.append(
            {
                "beat": idx + 1,
                "speech_seconds": round(len(samples) / RATE, 2),
                "tempo": round(tempo, 3),
            }
        )
    all_samples = np.concatenate(pieces)
    rms = float(np.sqrt(np.mean(all_samples**2)))
    gain = min(
        (32768 * 10 ** (-18 / 20)) / max(rms, 1),
        28500 / max(np.max(np.abs(all_samples)), 1),
    )
    with wave.open(str(folder / "narration.wav"), "wb") as wav:
        wav.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
        wav.writeframes(np.round(all_samples * gain).astype("<i2").tobytes())
    return receipts


def recipe_digest(item, folder, clock, beats):
    recipe = {
        "beats": beats,
        "voice": item["voice"],
        "timeline": clock,
        "sources": item["gallery"],
        "takes": [
            fingerprint(folder / f"creator-{k + 1}" / "raw.mp4")
            for k in range(sum(host for _, host in clock))
        ],
    }
    return hashlib.sha256(
        json.dumps(recipe, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def assemble(item, folder, index):
    from .campaign import final_script, verify_claims

    copy = final_script(item, folder, index)
    clock = timeline(index, item)
    copy_digest = recipe_digest(item, folder, clock, copy["beats"])
    dst = folder / "master.mp4"
    if (
        dst.is_file()
        and (folder / "edit.json").is_file()
        and read(folder / "edit.json").get("copy_sha256") == copy_digest
        and read(folder / "edit.json").get("edit_version") == 2
    ):
        return
    beats = copy["beats"]
    source = folder / "source"
    scenes = iter(item["gallery"])
    cues = []
    host = 0
    first = 0
    original_lines = [b["voice"] for b in beats]
    voice_data = narration(folder, item, beats, clock)
    spoken_lines = [b["voice"] for b in beats]
    if spoken_lines != original_lines:
        problems = verify_claims(item, spoken_lines)
        if problems:
            raise ValueError(
                "Shortened narration needs review: " + " / ".join(problems)
            )
    tmp = folder / "master.partial.mp4"
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "bgr24",
            "-video_size",
            f"{W}x{H}",
            "-framerate",
            str(FPS),
            "-i",
            "pipe:0",
            "-i",
            str(folder / "narration.wav"),
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-c:v",
            "libx264",
            "-threads",
            "4",
            "-preset",
            "veryfast",
            "-crf",
            "17",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-movflags",
            "+faststart",
            str(tmp),
        ],
        stdin=subprocess.PIPE,
    )
    try:
        for n, ((seconds, is_host), beat) in enumerate(zip(clock, beats, strict=True)):
            count = seconds * FPS
            if is_host:
                host += 1
                path = folder / f"creator-{host}" / "raw.mp4"
                frames = moving_frames(path, count, creator=True)
            else:
                spec = next(scenes)
                if "preview" in spec or "preview_at" in spec or "video" in spec:
                    path = (
                        source / spec["video"]
                        if "video" in spec
                        else preview_cfr(source)
                    )
                    frames = moving_frames(
                        path,
                        count,
                        spec.get("preview"),
                        allow_loop=spec.get("loop", False),
                        at=spec.get("preview_at"),
                    )
                else:
                    path = source / (
                        spec["file"]
                        if "file" in spec
                        else f"src-{spec['photo']:02d}.jpg"
                    )
                    frames = photo_frames(
                        path,
                        spec.get("box"),
                        count,
                        folder.parent / "models" / "depth-small.onnx",
                        1 if n % 2 else -1,
                    )
            target = folder / f"caption-{n}.png"
            # Caption the actual take, including a line shortened to fit its beat.
            plate(beat["voice"], (W, H), target, True)
            with Image.open(target) as im:
                bounds = im.getbbox()
                overlay = np.array(im.crop(bounds))
            x, y, right, bottom = bounds
            alpha = overlay[:, :, 3:4].astype(np.float32) / 255
            label = overlay[:, :, :3][:, :, ::-1] * alpha
            for frame in frames:
                result = frame.copy()
                result[y:bottom, x:right] = np.round(
                    label + result[y:bottom, x:right] * (1 - alpha)
                ).astype(np.uint8)
                process.stdin.write(result.tobytes())
            cues.append(
                {
                    "beat": n + 1,
                    "start_frame": first,
                    "end_frame": first + count,
                    "source": str(path.relative_to(folder)),
                    "creator": is_host,
                    **beat,
                    "caption": beat["voice"],
                }
            )
            first += count
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("FFmpeg campaign encode failed")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if not process.stdin.closed:
            process.stdin.close()
    tmp.replace(dst)
    save(
        folder / "edit.json",
        {
            "timeline": cues,
            "voice": item["voice"],
            "speech": voice_data,
            "sha256": fingerprint(dst),
            "copy_sha256": copy_digest,
            "edit_version": 2,
        },
    )


def caption_rasters(frame, path):
    (x, y, r, b), mask, _, _ = glyph_template(str(path))
    rows = np.where(mask.any(axis=1))[0]
    groups = np.split(rows, np.where(np.diff(rows) > 5)[0] + 1)
    crops = []
    for group in groups:
        ax = np.where(mask[group].any(axis=0))[0]
        crop = frame[
            y + group.min() - 2 : y + group.max() + 3,
            x + ax.min() - 2 : x + ax.max() + 3,
        ]
        bw = (cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) > 130).astype(np.uint8) * 255
        crops.append(
            cv2.copyMakeBorder(255 - bw, 4, 4, 5, 5, cv2.BORDER_CONSTANT, value=255)
        )
    return crops


class ShotReview(BaseModel):
    observation: str = Field(max_length=260)
    defect: Literal[
        "none", "extra_limb", "malformed_face", "fused_digits", "prominent_gibberish"
    ]
    location: str = Field(max_length=120)


def review_creator(folder, cue):
    path = folder / cue["source"]
    with FileLock(str(path.parent / "visual-review.lock"), timeout=1800):
        return inspect_creator(path, cue)


def inspect_creator(path, cue):
    key = fingerprint(path)
    dest = path.parent / "visual-review.json"
    if dest.is_file() and read(dest).get("sha256") == key:
        return read(dest)
    cap = cv2.VideoCapture(str(path))
    views = []
    try:
        for position in (0.18, 0.52, 0.85):
            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                round((cue["end_frame"] - cue["start_frame"] - 1) * position),
            )
            ok, frame = cap.read()
            if not ok:
                raise ValueError("Incomplete raw creator shot")
            views.append(cv2.resize(frame, (272, 484), interpolation=cv2.INTER_AREA))
    finally:
        cap.release()
    image = cv2.imencode(".jpg", np.hstack(views))[1].tobytes()
    review = ask(
        ShotReview,
        "Inspect three successive views of the SAME smartphone shot. Describe only visible content in one sentence. "
        "For defect choose none unless the images visibly show one of the listed severe artifacts. A partially hidden hand, "
        "motion blur or distant unreadable text is normal. If an artifact is clearly present, identify its exact visible location. "
        "With no defect return an empty location.",
        {"task": "Check visual anatomy and prominent synthesized lettering."},
        image,
    )
    result = {"sha256": key, **review.model_dump()}
    save(dest, result)
    return result


def verify(item, folder):
    master = folder / "master.mp4"
    digest = fingerprint(master)
    report = folder / "assessment.json"
    if (
        report.is_file()
        and read(report).get("sha256") == digest
        and read(report).get("passed")
        and (
            not item.get("asr_model")
            or read(report).get("transcription", {}).get("status") == "assessed"
        )
    ):
        return
    edit = read(folder / "edit.json")
    cues = edit["timeline"]
    issues = {}
    samples = []
    idx = 0
    blank = 0
    longest_freeze = 0
    last = None
    freeze = 0
    min_glyph = 1.0
    min_ocr = 1.0
    cache = {}
    ledger = folder / "frames.jsonl.gz"

    def problem(key):
        issues[key] = issues.get(key, 0) + 1

    with (
        av.open(str(master)) as video,
        gzip.open(ledger, "wt", encoding="utf-8") as rows,
    ):
        vs = video.streams.video[0]
        sound = video.streams.audio[0]
        fmt = [vs.width, vs.height, float(vs.average_rate)]
        video_duration = float(vs.duration * vs.time_base)
        audio_duration = float(sound.duration * sound.time_base)
        n = 0
        for decoded in video.decode(vs):
            while idx + 1 < len(cues) and n >= cues[idx]["end_frame"]:
                idx += 1
                last = None
                freeze = 0
            cue = cues[idx]
            frame = decoded.to_ndarray(format="bgr24")
            path = folder / f"caption-{idx}.png"
            glyph = glyph_quality(frame, path)
            min_glyph = min(glyph, min_glyph)
            rasters = caption_rasters(frame, path)
            key = (
                idx,
                tuple(hashlib.blake2b(r, digest_size=12).digest() for r in rasters),
            )
            if key not in cache:
                recognized, _ = recognizer().text_rec(
                    [cv2.cvtColor(r, cv2.COLOR_GRAY2BGR) for r in rasters]
                )
                cache[key] = " ".join(t for t, _ in recognized)
            actual = cache[key]
            score = similarity(cue["caption"], actual)
            min_ocr = min(score, min_ocr)
            flags = []
            if glyph < 0.82:
                flags.append("glyphs")
            if score < 0.86:
                flags.append("caption_ocr")
            if abs(float(decoded.time) - n / FPS) > 0.002:
                flags.append("timestamp")
            thumb = cv2.resize(
                cv2.cvtColor(frame[: int(0.64 * H)], cv2.COLOR_BGR2GRAY),
                (100, 108),
                interpolation=cv2.INTER_AREA,
            )
            is_blank = thumb.std() < 2.0 and (thumb.mean() < 8 or thumb.mean() > 248)
            blank = blank + 1 if is_blank else 0
            if blank >= FPS // 2:
                flags.append("blank")
            if last is not None:
                diff = np.mean(cv2.absdiff(last, thumb))
                freeze = freeze + 1 if diff < 0.035 else 0
                longest_freeze = max(longest_freeze, freeze)
                if cue["creator"] and freeze >= FPS:
                    flags.append("frozen_creator")
            last = thumb
            if n - cue["start_frame"] in [
                int((cue["end_frame"] - cue["start_frame"]) * v) for v in (0.22, 0.72)
            ]:
                samples.append(
                    cv2.resize(frame, (174, 310), interpolation=cv2.INTER_AREA)
                )
            for flag in flags:
                problem(flag)
            rows.write(
                json.dumps(
                    {
                        "frame": n,
                        "time": round(float(decoded.time), 5),
                        "beat": idx + 1,
                        "read": actual,
                        "ocr": round(score, 3),
                        "glyphs": round(glyph, 3),
                        "flags": flags,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n += 1
    if (
        n != FPS * 30
        or fmt != [W, H, FPS]
        or abs(video_duration - 30) > 0.001
        or abs(audio_duration - 30) > 0.025
    ):
        problem("delivery_format")
    audio = audio_metrics(master, True)
    if audio.get("mean_volume_db", -100) < -29 or audio.get("max_volume_db", 0) > -0.2:
        problem("audio_level")
    speech = {"status": "disabled"}
    if item.get("asr_model"):
        speech = transcript(master, folder.parent / item["asr_model"], "tr")
        if speech["status"] == "assessed":
            heard = " ".join(s["text"] for s in speech["segments"])
            expected = " ".join(c["voice"] for c in cues)
            speech["similarity"] = round(similarity(expected, heard), 3)
        if speech.get("similarity", 0) < 0.56:
            problem("speech_review")
    if len(samples) != 12:
        raise ValueError("Incomplete visual contact sheet")
    sheet = np.vstack([np.hstack(samples[k : k + 4]) for k in range(0, 12, 4)])
    cv2.imwrite(str(folder / "contact.jpg"), sheet)
    reviews = {c["beat"]: review_creator(folder, c) for c in cues if c["creator"]}
    failed_hosts = [num for num, rv in reviews.items() if rv["defect"] != "none"]
    if failed_hosts:
        problem("creator_review")
    result = {
        "passed": not issues,
        "sha256": digest,
        "frames": n,
        "publisher_frames": sum(
            c["end_frame"] - c["start_frame"] for c in cues if not c["creator"]
        ),
        "generated_creator_frames": sum(
            c["end_frame"] - c["start_frame"] for c in cues if c["creator"]
        ),
        "video_seconds": video_duration,
        "audio_seconds": audio_duration,
        "format": fmt,
        "caption_min_glyph": round(min_glyph, 3),
        "caption_min_ocr": round(min_ocr, 3),
        "max_nearly_static_frames": longest_freeze,
        "audio": audio,
        "transcription": speech,
        "issues": issues,
        "creator_review": reviews,
        "failed_creator_beats": failed_hosts,
        "vision_model": os.getenv("FRAMECRAFT_VISION_MODEL"),
        "speech": edit["speech"],
    }
    save(report, result)
    if issues:
        raise ValueError(f"Delivery checks failed: {issues}")


def sources_used(folder, cues, source, supplemental):
    photos = {p["file"]: p["url"] for p in source["photos"]}
    shots = []
    for cue in cues:
        path = folder / cue["source"]
        shot = {"start_s": cue["start_frame"] / FPS, "end_s": cue["end_frame"] / FPS}
        if cue["creator"]:
            shot.update(read(path.parent / "receipt.json"))
        elif path.name in {"preview.mp4", "preview-cfr.mp4"}:
            shot["source"] = source["preview_url"]
        else:
            key = path.relative_to(folder.parent).as_posix()
            shot["source"] = (
                photos.get(path.name) or supplemental.get(key) or source["url"]
            )
        shots.append(shot)
    return shots


def export_set(root):
    items = read(root / "campaign.json")["items"]
    out = root / "delivery"
    out.mkdir(exist_ok=True)
    cards = []
    catalog = []
    supplemental_path = root / "supplementary-sources.json"
    supplemental = read(supplemental_path) if supplemental_path.is_file() else {}
    for index, item in enumerate(items):
        folder = root / item["slug"]
        qc = read(folder / "assessment.json")
        if not qc["passed"] or fingerprint(folder / "master.mp4") != qc["sha256"]:
            raise ValueError(f"Not approved: {item['slug']}")
        copy = read(folder / "studio-copy.json")
        if (
            item.get("editor_model") and copy.get("editor") != item["editor_model"]
        ) or not copy.get("claims_checked"):
            raise ValueError(f"Unverified advertising copy: {item['slug']}")
        if read(folder / "edit.json")["copy_sha256"] != recipe_digest(
            item, folder, timeline(index, item), copy["beats"]
        ):
            raise ValueError(
                f"Render is older than its approved ingredients: {item['slug']}"
            )
    for item in items:
        slug = item["slug"]
        folder = root / slug
        qc = read(folder / "assessment.json")
        src = read(folder / "source" / "source.json")
        name = slug + ".mp4"
        shutil.copyfile(folder / "master.mp4", out / name)
        shutil.copyfile(folder / "frames.jsonl.gz", out / (slug + "-frames.jsonl.gz"))
        cap = cv2.VideoCapture(str(out / name))
        try:
            cues = read(folder / "edit.json")["timeline"]
            cue = next(c for c in cues if not c["creator"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, cue["start_frame"] + FPS)
            ok, img = cap.read()
            if not ok:
                raise ValueError("Unusable preview")
            cv2.imwrite(str(out / (slug + ".jpg")), cv2.resize(img, (360, 640)))
        finally:
            cap.release()
        rec = {
            "name": item["name"],
            "audience": item["audience"],
            "url": src["url"],
            "source_facts": item["proof"],
            "file": name,
            "captions": [b["caption"] for b in cues],
            "voiceover": [b["voice"] for b in cues],
            "voice_model": f"Piper tr_TR-{item['voice']}-medium",
            "shots": sources_used(folder, cues, src, supplemental),
            "qa": qc,
        }
        catalog.append(rec)
        cards.append(
            f'<article><video controls preload="none" playsinline poster="{slug}.jpg" src="{name}"></video><h2>{html.escape(item["name"])}</h2><p>{html.escape(item["audience"])}</p><a href="{html.escape(src["url"], quote=True)}">Ürün sayfası</a> · <a href="{name}" download>MP4 indir</a></article>'
        )
    save(out / "rapor.json", catalog)
    (out / "index.html").write_text(
        '<!doctype html><html lang="tr"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>UGC Creator · UGC seçkisi</title><style>body{font:16px system-ui;background:#f6f3ed;color:#202427;margin:clamp(1rem,4vw,4rem)}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:2rem}video{display:block;width:100%;aspect-ratio:9/16;background:#181b1e;border-radius:12px}h2{font-size:1.1rem;margin-bottom:.35rem}p{margin-top:0;color:#50575b}a{color:#1b505b}header{margin-bottom:2rem}</style><header><h1>UGC seçkisi</h1><p>Türkçe seslendirme · 30 saniye · 9:16</p></header><main>'
        + "".join(cards)
        + "</main></html>",
        encoding="utf-8",
    )
    bundle = root / f"Framecraft-UGC-{len(items)}.zip"
    with zipfile.ZipFile(
        bundle.with_suffix(".partial"), "w", compression=zipfile.ZIP_STORED
    ) as archive:
        for asset in sorted(out.iterdir()):
            if asset.is_file():
                archive.write(asset, asset.name)
    bundle.with_suffix(".partial").replace(bundle)


def process(stage, item, folder, index):
    if stage == "assemble":
        assemble(item, folder, index)
    elif stage == "verify":
        verify(item, folder)


def finish_item(item, folder, index):
    with FileLock(str(folder / "finish.lock"), timeout=3600):
        finish_locked(item, folder, index)


def finish_locked(item, folder, index):
    from .director import rewrite_shot

    repairs = folder / "campaign-repairs.json"
    history = read(repairs) if repairs.is_file() else []
    while True:
        assemble(item, folder, index)
        try:
            verify(item, folder)
            return
        except ValueError:
            if not (folder / "assessment.json").is_file():
                raise
            qc = read(folder / "assessment.json")
            if set(qc["issues"]) != {"creator_review"} or len(history) >= 4:
                raise
        cues = read(folder / "edit.json")["timeline"]
        for number in qc["failed_creator_beats"]:
            cue = cues[number - 1]
            dest = folder / cue["source"]
            receipt = read(dest.parent / "receipt.json")
            previous = receipt["prompt"]
            visual = rewrite_shot(
                {
                    "visual": previous,
                    "checks": [
                        item["audience"],
                        "natural movement, plausible anatomy, unlettered surroundings",
                    ],
                },
                previous,
                [qc["creator_review"][str(number)]],
            )
            seed = secrets.randbelow(2**31 - 1)
            tmp = dest.parent / "replacement.mp4"
            generate("h3", visual, seed, "vertical", tmp, print, frames=97)
            history.append(
                {
                    "beat": number,
                    "previous": receipt,
                    "evidence": qc["creator_review"],
                    "new_seed": seed,
                }
            )
            save(repairs, history)
            tmp.replace(dest)
            save(
                dest.parent / "receipt.json",
                {
                    "provider": "h3",
                    "seed": seed,
                    "prompt": visual,
                    "frames_requested": 97,
                },
            )
        (folder / "edit.json").unlink()
