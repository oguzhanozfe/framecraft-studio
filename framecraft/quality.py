import re
import subprocess
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image

from .core import setting


@lru_cache(maxsize=2)
def spelling_dictionary(language):
    from phunspell import Phunspell

    return Phunspell({"tr": "tr_TR", "en": "en_US"}[language])


def caption_spelling(captions, language, vocabulary=""):
    pattern = r"[^\W\d_]+(?:['’][^\W\d_]+)*"

    def normalize(word):
        return (
            word.replace("I", "ı").replace("İ", "i") if language == "tr" else word
        ).lower()

    allowed = {normalize(w) for w in re.findall(pattern, vocabulary)}
    dictionary = spelling_dictionary(language)
    issues = []
    for cue, caption in enumerate(captions, 1):
        for word in dict.fromkeys(re.findall(pattern, caption)):
            lower = normalize(word)
            if (
                len(word) > 1
                and not word.isupper()
                and lower not in allowed
                and not dictionary.lookup(word)
                and not dictionary.lookup(lower)
            ):
                issues.append(f"Cue {cue} · {word}")
    return issues


@lru_cache(maxsize=8)
def glyph_template(path):
    with Image.open(path) as im:
        area = im.getbbox()
        crop = np.asarray(im.crop(area))
    mask = (crop[:, :, 0] > 190) & (crop[:, :, 3] > 190)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype("uint8"))
    return area, mask, labels, stats[1:, cv2.CC_STAT_AREA]


def glyph_quality(bgr, path):
    area, mask, labels, areas = glyph_template(str(path))
    x, y, r, b = area
    crop = bgr[y:b, x:r]
    if crop.shape[:2] != mask.shape:
        return 0.0
    seen = crop.min(axis=2) > 175
    hits = np.bincount(labels[mask & seen], minlength=len(areas) + 1)[1:]
    glyph_hits = hits[areas >= 4] / areas[areas >= 4]
    precision = np.count_nonzero(mask & seen) / max(1, np.count_nonzero(seen))
    return float(min(precision, min(glyph_hits, default=0.0)))


def audio_metrics(video, has_audio):
    if not has_audio:
        return {"stream": False}
    run = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-i",
            str(video),
            "-vn",
            "-af",
            "volumedetect",
            "-f",
            "null",
            "-",
        ],
        stderr=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        timeout=600,
    )
    txt = run.stderr.decode(errors="replace")
    data = {"stream": True}
    for key in ("mean_volume", "max_volume"):
        match = re.search(key + r": ([-\d.]+) dB", txt)
        if match:
            data[key + "_db"] = float(match[1])
    return data


def transcript(video, model=None, language=None):
    model = model or setting("TRANSCRIBE_MODEL")
    if not model:
        return {"status": "disabled"}
    try:
        from faster_whisper import WhisperModel

        audio = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-nostdin",
                "-i",
                str(video),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-f",
                "f32le",
                "pipe:1",
            ],
            capture_output=True,
            check=True,
            timeout=600,
        )
        segments, info = WhisperModel(
            str(model), device="cpu", compute_type="int8", cpu_threads=4
        ).transcribe(
            np.frombuffer(audio.stdout, dtype="<f4"), language=language, vad_filter=True
        )
        return {
            "status": "assessed",
            "language": info.language,
            "segments": [
                {
                    "start": round(s.start, 2),
                    "end": round(s.end, 2),
                    "text": s.text.strip(),
                }
                for s in segments
            ],
        }
    except (
        ImportError,
        OSError,
        RuntimeError,
        ValueError,
        subprocess.SubprocessError,
    ) as exc:
        return {"status": "unavailable", "reason": str(exc)[:250]}
