"""Produce motion-only campaign cuts from timed footage and an unretimed local narration."""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import av
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from filelock import FileLock

from .audit import read_text, recognizer, similarity
from .core import read, save
from .quality import audio_metrics, transcript

FPS = 30
SIZE = (1080, 1920)


def run(args):
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", *map(str, args)], check=True
    )


def target_frames(recipe):
    seconds = recipe.get("preview_seconds", 30)
    frames = round(seconds * FPS)
    if not 1 <= frames <= 30 * FPS or abs(frames / FPS - seconds) > 0.0001:
        raise ValueError("Preview duration must be a whole number of frames within 30s")
    return frames


def spans(recipe, folder):
    seen = {}
    total = 0
    for shot in recipe["shots"]:
        length = round(shot["seconds"] * FPS)
        total += length
        sources = [shot] + ([shot["inset"]] if shot.get("inset") else [])
        for view in sources:
            path = (folder / view["file"]).resolve()
            start = view.get("start", 0)
            if (
                path.suffix.lower() not in {".mp4", ".mov", ".mkv", ".webm"}
                or length < 1
                or start < 0
            ):
                raise ValueError(f"Motion footage required: {path.name}")
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                duration = (
                    float(stream.duration * stream.time_base)
                    if stream.duration
                    else container.duration / av.time_base
                )
            stop = start + length / FPS
            if stop > duration + 0.025:
                raise ValueError(f"{path.name} is too short for this shot")
            previous = seen.setdefault(path, [])
            if any(start < b - 0.015 and stop > a + 0.015 for a, b in previous):
                raise ValueError(f"Repeated footage: {path.name} at {start}s")
            previous.append((start, stop))
    if total != target_frames(recipe):
        raise ValueError(
            f"Campaign needs exactly {target_frames(recipe) / FPS:g}s, supplied {total / FPS:.3f}s"
        )


def font(size, bold=False):
    custom = os.getenv("FRAMECRAFT_FONT_BOLD" if bold else "FRAMECRAFT_FONT")
    candidates = [
        custom,
        "segoeuib.ttf" if bold else "segoeui.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for f in filter(None, candidates):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    raise RuntimeError(
        "Install a font with Turkish glyphs, or set FRAMECRAFT_FONT and FRAMECRAFT_FONT_BOLD."
    )


def graphic(shot, path):
    w, h = SIZE
    image = Image.new("RGBA", SIZE)
    draw = ImageDraw.Draw(image)
    rows = []
    layout = shot.get("layout", "lower")
    if layout == "clean":
        image.save(path)
        return rows
    x = 74
    fields = [
        ("eyebrow", 1335, 35, False),
        ("headline", 1400, 91, True),
        ("detail", 1525, 41, False),
        ("credit", 1613, 29, False),
    ]
    if layout == "caption":
        fields = [
            ("headline", 1384, 54, True),
            ("detail", 1467, 43, False),
            ("credit", 1534, 32, False),
        ]
    elif layout == "corner":
        line = shot.get("headline", "")
        fields = [("headline", 253, 37, True)]
        span = round(draw.textlength(line, font=font(37, True)))
        draw.rounded_rectangle(
            (49, 230, 105 + span, 325), radius=20, fill=(13, 11, 24, 226)
        )
        draw.rounded_rectangle((55, 247, 61, 306), radius=3, fill=(140, 245, 214, 255))
    elif layout == "versus":
        fields = [
            ("eyebrow", 248, 38, True),
            ("headline", 312, 82, True),
            ("detail", 490, 66, True),
            ("credit", 662, 29, False),
        ]
        for col in range(w):
            alpha = round(235 * max(0.0, min(1.0, (w - col) / 280)))
            draw.line((col, 190, col, 742), fill=(9, 10, 22, alpha))
        draw.rectangle((74, 219, 181, 225), fill=(119, 245, 215, 255))
    elif layout == "metric":
        fields = [
            ("eyebrow", 230, 40, True),
            ("headline", 296, 152, True),
            ("detail", 500, 47, False),
            ("credit", 587, 32, False),
        ]
        # Keep the right of frame free for the presenter or the action.
        for col in range(w):
            alpha = round(232 * max(0, 1 - max(0, col - 570) / 510))
            draw.line((col, 160, col, 690), fill=(9, 10, 22, alpha))
        draw.rectangle((74, 205, 184, 212), fill=(236, 95, 174, 255))
    elif layout == "tag":
        fields = [
            ("eyebrow", 230, 33, True),
            ("headline", 293, 60, True),
            ("detail", 377, 32, False),
            ("credit", 432, 26, False),
        ]
        for y in range(100, 550):
            fade = min(1.0, (y - 100) / 100, (550 - y) / 90)
            draw.line((0, y, w * 0.82, y), fill=(10, 10, 20, round(225 * fade)))
    elif layout == "side":
        x = 58
        fields = [
            ("eyebrow", 1030, 34, True),
            ("headline", 1090, 85, True),
            ("detail", 1205, 37, False),
            ("credit", 1270, 28, False),
        ]
        for col in range(w):
            alpha = round(232 * max(0.0, min(1.0, (940 - col) / 290)))
            draw.line((col, 985, col, 1350), fill=(8, 11, 21, alpha))
        draw.rectangle((58, 1001, 158, 1007), fill=(107, 240, 222, 255))
    else:
        for y in range(940, h):
            t = min(1.0, max(0.0, (y - 940) / 480))
            draw.line((0, y, w, y), fill=(6, 10, 15, round(239 * t)))
    for key, y, size, bold in fields:
        line = shot.get(key, "").strip()
        if not line:
            continue
        face = font(size, bold)
        while draw.textlength(line, font=face) > w - x - 133 and size > 23:
            size -= 1
            face = font(size, bold)
        if draw.textlength(line, font=face) > w - x - 133:
            raise ValueError(f"Text outside the safe area: {line}")
        # All glyphs remain unbroken and in a stationary safe area for the shot.
        color = (
            (255, 255, 255, 255)
            if key not in ("eyebrow", "credit")
            else (216, 232, 235, 255)
        )
        if layout == "metric" and key == "headline":
            color = (185, 255, 234, 255)
        box = draw.textbbox((x, y), line, font=face)
        if layout == "caption":
            draw.rounded_rectangle(
                (box[0] - 14, box[1] - 8, box[2] + 14, box[3] + 8),
                radius=7,
                fill=(15, 15, 16, 237),
            )
        draw.text((x, y), line, font=face, fill=color)
        rows.append(
            {"text": line, "box": [box[0] - 2, box[1] - 2, box[2] + 2, box[3] + 2]}
        )
    image.save(path)
    return rows


def place_footage(shots, folder):
    """Move supplied trailer spans away from slates/black fades before a paid creative is encoded."""
    shots = [s.copy() for s in shots]
    for shot in shots:
        radius = shot.get("search_radius", 0)
        if not radius:
            continue
        cap = cv2.VideoCapture(str(folder / shot["file"]))
        titles = {}
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
            seconds = shot["seconds"]
            origin = shot.get("start", 0)
            others = [
                (s.get("start", 0), s.get("start", 0) + s["seconds"])
                for s in shots
                if s is not shot and s["file"] == shot["file"]
            ]
            shifts = [0] + [
                d
                for step in range(1, round(radius * 2) + 1)
                for d in (step / 2, -step / 2)
            ]
            for shift in shifts:
                start = origin + shift
                if (
                    start < 0
                    or start + seconds > total + 0.01
                    or any(
                        start < b - 0.01 and start + seconds > a + 0.01
                        for a, b in others
                    )
                ):
                    continue
                cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
                for i in range(round(seconds * fps)):
                    ok, image = cap.read()
                    if not ok:
                        break
                    h, w = image.shape[:2]
                    if (
                        shot.get("avoid_title_cards")
                        and i % max(1, round(fps / 3)) == 0
                    ):
                        frame_no = round(start * fps) + i
                        if frame_no not in titles:
                            titles[frame_no] = any(
                                line["confidence"] >= 0.75
                                and len(line["text"].strip()) >= 4
                                and line["box"][3] - line["box"][1] > 0.06
                                for line in read_text(image)
                            )
                        if titles[frame_no]:
                            break
                    strip = int(min(w, h * 9 / 16))
                    x = round((w - strip) * shot.get("horizontal_focus", 0.5))
                    pixels = cv2.resize(
                        image[h // 8 : h * 2 // 3, x : x + strip], (32, 48)
                    )
                    if pixels.std() < 3 and (pixels.mean() < 12 or pixels.mean() > 245):
                        break
                else:
                    shot["start"] = start
                    break
            else:
                raise ValueError("No continuous visible span near " + shot["file"])
        finally:
            cap.release()
    return shots


def render_soundtrack(recipe, folder, soundtrack, seconds):
    voice = folder / recipe["voice"]
    inputs = ["-i", voice]
    speech = f"loudnorm=I=-16:TP=-1.6:LRA=9,aresample=48000,adelay=120,apad,atrim=duration={seconds}"
    if recipe.get("score"):
        inputs += ["-i", folder / recipe["score"]]
        mix = (
            f"[0:a]{speech},aformat=channel_layouts=stereo,asplit=2[vo][key];"
            f"[1:a]loudnorm=I=-24:TP=-3:LRA=8,aresample=48000,afade=t=out:st={max(0, seconds - 1)}:d=1[bed];"
            "[bed][key]sidechaincompress=threshold=0.024:ratio=3:attack=12:release=260[under];"
            "[vo][under]amix=inputs=2:normalize=0:duration=first,alimiter=limit=0.90:latency=1[a]"
        )
        audio = ["-filter_complex", mix, "-map", "[a]"]
    else:
        audio = ["-map", "0:a", "-af", speech]
    run(
        [
            *inputs,
            *audio,
            "-t",
            str(seconds),
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            soundtrack,
        ]
    )


def render(recipe, folder):
    folder = Path(folder)
    recipe = {**recipe, "shots": place_footage(recipe["shots"], folder)}
    spans(recipe, folder)
    seconds = target_frames(recipe) / FPS
    output = folder / recipe.get("output", "motion.mp4")
    with tempfile.TemporaryDirectory(dir=folder) as tmp:
        scratch = Path(tmp)
        parts = []
        cues = []
        index = 0
        for n, shot in enumerate(recipe["shots"]):
            overlay = folder / f"motion-text-{n:02}.png"
            lines = graphic(shot, overlay)
            frames = round(shot["seconds"] * FPS)
            framing = float(shot.get("horizontal_focus", 0.5))
            if not 0 <= framing <= 1:
                raise ValueError("horizontal_focus must be in [0,1]")
            if shot.get("framing") == "contain":
                scale = "scale=1080:1320:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(1320-ih)/2+95:color=0x10151B"
            else:
                scale = f"scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920:(iw-ow)*{framing}:(ih-oh)/2"
            vf = f"[0:v]setpts=PTS-STARTPTS,fps={FPS},{scale},setsar=1[base];"
            inset = shot.get("inset")
            extra = []
            layer = "base"
            if inset:
                iw, ih = inset.get("width", 352), inset.get("height", 466)
                ix, iy = inset.get("x", 652), inset.get("y", 770)
                extra = [
                    "-ss",
                    str(inset.get("start", 0)),
                    "-i",
                    folder / inset["file"],
                ]
                vf += f"[2:v]setpts=PTS-STARTPTS,fps={FPS},scale={iw}:{ih}:force_original_aspect_ratio=increase,crop={iw}:{ih},setsar=1,pad=iw+6:ih+6:3:3:color=0x95ead4[inset];[base][inset]overlay={ix}:{iy}:eof_action=pass[duet];"
                layer = "duet"
            vf += f"[{layer}][1:v]overlay=0:0:format=auto[texted];"
            layer = "texted"
            if shot.get("layout") == "versus":
                values = shot["comparison"]
                if len(values) != 2 or min(values) <= 0:
                    raise ValueError("Two positive benchmark values required")
                for row, (val, color, top) in enumerate(
                    zip(values, ("0x82f2d0", "0xf887bf"), (439, 609))
                ):
                    ratio = val / max(values)
                    vf += (
                        f"color=c=0x32404a:s=500x10:r={FPS}:d={shot['seconds']}[track{row}];"
                        f"color=c={color}:s=500x10:r={FPS}:d={shot['seconds']}[fill{row}];"
                        f"[track{row}][fill{row}]overlay=x='-w+w*{ratio}*min(1,t/0.38)':y=0[bar{row}];"
                        f"[{layer}][bar{row}]overlay=74:{top}[withbar{row}];"
                    )
                    layer = f"withbar{row}"
            vf += f"[{layer}]format=yuv420p[out]"
            part = scratch / f"part-{n:02}.mp4"
            run(
                [
                    "-ss",
                    str(shot.get("start", 0)),
                    "-i",
                    folder / shot["file"],
                    "-i",
                    overlay,
                    *extra,
                    "-filter_complex_threads",
                    "3",
                    "-filter_complex",
                    vf,
                    "-map",
                    "[out]",
                    "-an",
                    "-frames:v",
                    str(frames),
                    "-c:v",
                    "libx264",
                    "-threads",
                    "4",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "17",
                    part,
                ]
            )
            parts.append(part)
            cues.append(
                {
                    "from": index,
                    "to": index + frames,
                    "plate": overlay.name,
                    "lines": lines,
                    "source": shot["file"],
                    "source_start_seconds": shot.get("start", 0),
                }
            )
            index += frames
        listing = scratch / "parts.txt"
        listing.write_text(
            "".join(f"file '{p.name}'\n" for p in parts), encoding="utf-8"
        )
        final = scratch / "master.mp4"
        streams = ["-map", "0:v", "-an"]
        if not recipe.get("silent"):
            soundtrack = scratch / "soundtrack.m4a"
            render_soundtrack(recipe, folder, soundtrack, seconds)
            streams = ["-i", soundtrack, "-map", "0:v", "-map", "1:a", "-c:a", "copy"]
        run(
            [
                # A change in source color metadata must not reset the frame counter.
                "-reinit_filter",
                "0",
                "-f",
                "concat",
                "-safe",
                "1",
                "-i",
                listing,
                *streams,
                "-vf",
                "settb=1/90000,setpts=N*3000",
                "-r",
                "30",
                "-video_track_timescale",
                "90000",
                "-c:v",
                "libx264",
                "-threads",
                "4",
                "-preset",
                "veryfast",
                "-crf",
                "16",
                "-movflags",
                "+faststart",
                final,
            ]
        )
        report = audit(final, folder, cues, recipe)
        if not report["passed"]:
            raise ValueError("Motion QA: " + ", ".join(report["failures"]))
        final.replace(output)
        if recipe.get("project"):
            from .projects import archive

            archive(
                recipe["project"],
                output,
                stage="preview" if recipe.get("preview_seconds") else "master",
                report=folder / "motion-assessment.json",
                ledger=folder / "motion-frames.jsonl.gz",
            )
    return output


def background_key(img):
    resized = cv2.resize(img, (24, 28), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)


def audit(path, folder, cues, recipe):
    all_failures = set()
    hashes = {}
    thumbnails = []
    plate_data = {}
    cache = {}
    samples = []
    idx = 0
    count = 0
    last = None
    near = 0
    sequence = []
    ledger = folder / "motion-frames.jsonl.gz"
    with av.open(str(path)) as video, gzip.open(ledger, "wt", encoding="utf-8") as log:
        v = video.streams.video[0]
        a = next(iter(video.streams.audio), None)
        silent = bool(recipe.get("silent"))
        av_length = float(a.duration * a.time_base) if a and a.duration else 0
        if silent and a is not None:
            all_failures.add("unexpected_audio")
        actual_rate = float(v.average_rate)
        if (
            (v.width, v.height) != SIZE
            or actual_rate != FPS
            or (not silent and abs(av_length - target_frames(recipe) / FPS) > 0.025)
        ):
            all_failures.add("format")
        for f in video.decode(v):
            if idx + 1 < len(cues) and count == cues[idx]["to"]:
                idx += 1
                last = None
                near = 0
                sequence = []
            cue = cues[idx]
            img = f.to_ndarray(format="bgr24")
            flags = set()
            observed = []
            if idx not in plate_data:
                plate_data[idx] = np.asarray(
                    Image.open(folder / cue["plate"]).convert("RGBA")
                )
            plate = plate_data[idx]
            for i, row in enumerate(cue["lines"]):
                x, y, r, b = row["box"]
                crop = cv2.cvtColor(img[y:b, x:r], cv2.COLOR_BGR2GRAY)
                # With a dark scrim, thresholding actual decoded pixels isolates the lettering.
                binary = (crop > 172).astype("uint8") * 255
                target = plate[y:b, x:r]
                mask = (target[:, :, 0] > 180) & (target[:, :, 3] > 245)
                coverage = np.count_nonzero(mask & (binary > 0)) / max(
                    1, np.count_nonzero(mask)
                )
                key = (idx, i, hashlib.blake2b(binary, digest_size=10).digest())
                if key not in cache:
                    clean = cv2.copyMakeBorder(
                        255 - binary, 5, 5, 7, 7, cv2.BORDER_CONSTANT, value=255
                    )
                    labels, _ = recognizer().text_rec(
                        [cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR)]
                    )
                    cache[key] = labels[0][0] if labels else ""
                readback = cache[key]
                score = similarity(row["text"], readback)
                if score < 0.86 or coverage < 0.90:
                    flags.add("text")
                observed.append([readback, round(score, 3), round(coverage, 3)])
            thumb = background_key(img[150:1270])
            if last is not None:
                near = near + 1 if np.mean(cv2.absdiff(thumb, last)) < 0.18 else 0
                if near >= FPS // 2:
                    flags.add("frozen")
            last = thumb
            if float(thumb.std()) < 1.5:
                flags.add("empty")
            if count % 3 == 0:
                thumbnails.append(thumb)
                # Ignore isolated cut matches, flag repeated one-second sequences.
                sequence.append(
                    hashlib.blake2b((thumb // 10).tobytes(), digest_size=8).digest()
                )
                if len(sequence) >= 10:
                    key = b"".join(sequence[-10:])
                    if key in hashes:
                        flags.add("loop")
                    hashes[key] = count
            if abs(float(f.time) - count / FPS) > 0.003:
                flags.add("timestamp")
            if count in (cue["from"] + 3, (cue["from"] + cue["to"]) // 2):
                samples.append(
                    cv2.resize(img, (150, 267), interpolation=cv2.INTER_AREA)
                )
            all_failures.update(flags)
            log.write(
                json.dumps(
                    {
                        "frame": count,
                        "time": round(float(f.time), 4),
                        "shot": idx,
                        "text": observed,
                        "flags": sorted(flags),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1
    if count != target_frames(recipe):
        all_failures.add("frame_count")
    if repeated_motion(thumbnails):
        all_failures.add("repeated_motion")
    levels = {"stream": False} if silent else audio_metrics(path, True)
    if not silent and (
        levels.get("mean_volume_db", -99) < -29 or levels.get("max_volume_db", 0) > -0.2
    ):
        all_failures.add("audio_level")
    speech = {}
    if not silent and recipe.get("asr_model"):
        speech = transcript(path, folder / recipe["asr_model"], "tr")
        heard = " ".join(s["text"] for s in speech.get("segments", []))
        speech["similarity"] = round(similarity(recipe["spoken"], heard), 3)
        if speech["similarity"] < 0.70:
            all_failures.add("voice_transcript")
    if samples:
        while len(samples) % 4:
            samples.append(np.zeros_like(samples[0]))
        cv2.imwrite(
            str(folder / "motion-contact.jpg"),
            np.vstack(
                [np.hstack(samples[i : i + 4]) for i in range(0, len(samples), 4)]
            ),
        )
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    result = {
        "passed": not all_failures,
        "failures": sorted(all_failures),
        "frames": count,
        "fps": FPS,
        "audio_seconds": av_length,
        "video_fps": actual_rate,
        "audio": levels,
        "speech": speech,
        "sha256": digest,
        "source_evidence": recipe.get("evidence", []),
        "voice": recipe.get("voice_provider"),
        "timeline": cues,
    }
    save(folder / "motion-assessment.json", result)
    return result


def repeated_motion(frames):
    """Match one-second moving sequences even after re-encoding changes pixel values."""
    if len(frames) < 20:
        return False
    a = np.asarray(frames, dtype=np.int16).reshape(len(frames), -1)
    for end in range(19, len(a)):
        candidates = np.flatnonzero(np.abs(a[: end - 9] - a[end]).mean(axis=1) < 1.4)
        for prev in candidates[candidates >= 9]:
            recent = a[end - 9 : end + 1]
            if (
                np.abs(recent[-1] - recent[0]).mean() > 2
                and np.abs(recent - a[prev - 9 : prev + 1]).mean() < 1.4
            ):
                return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path)
    args = parser.parse_args()
    with FileLock(str(args.recipe.parent / "motion.lock"), timeout=3600):
        print(render(read(args.recipe), args.recipe.parent))


if __name__ == "__main__":
    main()
