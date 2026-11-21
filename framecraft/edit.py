import os
import subprocess

import av
from PIL import Image, ImageDraw, ImageFont

CANVAS = {"vertical": (1080, 1920), "square": (1080, 1080), "wide": (1920, 1080)}


def ffmpeg(args):
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            *map(str, args),
        ],
        capture_output=True,
        timeout=1200,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace")[-1500:])


def font_at(size):
    fonts = [
        os.environ.get("FRAMECRAFT_FONT"),
        "DejaVuSans-Bold.ttf",
        "segoeuib.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ]
    for name in filter(None, fonts):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    raise RuntimeError("Install DejaVu Sans or set FRAMECRAFT_FONT to a Unicode TTF")


def plate(text, size, path, repair=False):
    w, h = size
    layer = Image.new("RGBA", size)
    draw = ImageDraw.Draw(layer)
    scale = min(w, h)
    for px in range(round(scale * (0.057 if repair else 0.050)), 18, -2):
        font = font_at(px)
        lines = [""]
        for word in text.split():
            proposed = (lines[-1] + " " + word).strip()
            if lines[-1] and draw.textlength(proposed, font=font) > w * 0.70:
                lines.append(word)
            else:
                lines[-1] = proposed
        if len(lines) <= 2 and all(
            draw.textlength(l, font=font) <= w * 0.70 for l in lines
        ):
            break
    else:
        raise ValueError("Caption cannot fit the safe area")
    rendered = "\n".join(lines)
    box = draw.multiline_textbbox(
        (0, 0), rendered, font=font, spacing=6, align="center"
    )
    bw, bh = box[2] - box[0], box[3] - box[1]
    center = (w * 0.47, h * 0.70)
    x, y = int(center[0] - bw / 2), int(center[1] - bh / 2)
    pad = int(scale * 0.018)
    draw.rounded_rectangle(
        (x - pad, y - pad, x + bw + pad, y + bh + pad),
        radius=pad,
        fill=(15, 18, 22, 255 if repair else 227),
    )
    draw.multiline_text(
        (x - box[0], y - box[1]),
        rendered,
        font=font,
        fill="white",
        spacing=6,
        align="center",
    )
    layer.save(path)


def render(
    clips, captions, shape, folder, repair=False, preserve_source=False, silent=False
):
    size = CANVAS[shape]
    cues = []
    parts = []
    cursor = 0
    for i, (source, caption) in enumerate(zip(clips, captions, strict=True)):
        with av.open(str(source)) as c:
            has_audio = bool(c.streams.audio)
            frames = sum(1 for _ in c.decode(video=0))
            frame_rate = float(c.streams.video[0].average_rate or 24)
        duration = round(frames * 24 / frame_rate) / 24
        img = folder / f"caption-{i}.png"
        plate(caption, size, img, repair)
        part = folder / f"edit-{i}.mp4"
        w, h = size
        args = ["-i", source, "-i", img]
        if not has_audio and not silent:
            args += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
        framing = f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}"
        if preserve_source:
            framing = f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2"
        audio = (
            ["-an"]
            if silent
            else [
                "-map",
                "0:a:0" if has_audio else "2:a:0",
                "-af",
                "aresample=48000,apad",
                "-c:a",
                "aac",
                "-ac",
                "2",
                "-ar",
                "48000",
            ]
        )
        args += [
            "-filter_complex",
            f"[0:v]fps=24,{framing},setsar=1[b];[b][1:v]overlay=0:0[v]",
            "-map",
            "[v]",
            *audio,
            "-t",
            f"{duration:.6f}",
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
            part,
        ]
        ffmpeg(args)
        cues.append(
            {
                "start": round(cursor, 5),
                "end": round(cursor + duration, 5),
                "text": caption,
            }
        )
        cursor += duration
        parts.append(part)
    concat = folder / "clips.txt"
    concat.write_text("".join(f"file '{p.name}'\n" for p in parts), encoding="utf-8")
    out = folder / "master.mp4"
    ffmpeg(
        [
            "-f",
            "concat",
            "-safe",
            "1",
            "-i",
            concat,
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            out,
        ]
    )
    return out, cues
