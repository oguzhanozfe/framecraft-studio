import re
import subprocess
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image

from .core import setting


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


def transcript(video):
    model = setting("TRANSCRIBE_MODEL")
    if not model:
        return {"status": "disabled"}
    try:
        from faster_whisper import WhisperModel

        segments, info = WhisperModel(
            model, device="cpu", compute_type="int8"
        ).transcribe(str(video), vad_filter=True)
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
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        return {"status": "unavailable", "reason": str(exc)[:250]}
