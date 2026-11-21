import json
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

import av
import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

from .core import save
from .quality import audio_metrics, glyph_quality


@lru_cache(maxsize=1)
def recognizer():
    return RapidOCR(
        intra_op_num_threads=3,
        inter_op_num_threads=1,
        det_limit_side_len=736,
        det_limit_type="max",
        use_cls=False,
    )


def norm(value):
    value = unicodedata.normalize("NFKD", value.casefold().replace("ı", "i"))
    return "".join(c for c in value if c.isalnum())


def similarity(expected, seen):
    return SequenceMatcher(None, norm(expected), norm(seen), autojunk=False).ratio()


def read_text(img):
    h, w = img.shape[:2]
    ratio = min(1, 960 / max(w, h))
    small = (
        cv2.resize(img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
        if ratio < 1
        else img
    )
    lines, _ = recognizer()(small, use_cls=False)
    out = []
    for points, text, conf in lines or []:
        coords = np.array(points) / np.array([small.shape[1], small.shape[0]])
        box = [*coords.min(axis=0), *coords.max(axis=0)]
        out.append(
            dict(
                text=text,
                confidence=round(float(conf), 3),
                box=np.round(box, 4).tolist(),
            )
        )
    return out


def text_checks(lines, expected, mode):
    flags = []
    if expected:
        inside = [l for l in lines if 0.59 < (l["box"][1] + l["box"][3]) / 2 < 0.81]
        inside.sort(key=lambda l: (round(l["box"][1], 2), l["box"][0]))
        if similarity(expected, " ".join(l["text"] for l in inside)) < 0.88:
            flags.append("caption_mismatch")
        if any(l["box"][0] < 0.055 or l["box"][2] > 0.90 for l in inside):
            flags.append("caption_cropped")
    if mode in ("raw", "master"):
        for line in lines:
            if expected and 0.59 < (line["box"][1] + line["box"][3]) / 2 < 0.81:
                continue
            if line["confidence"] >= 0.74 and len(norm(line["text"])) >= 3:
                flags.append("unplanned_text")
                break
    return flags


def judge(counts, n):
    weights = {
        "black": 120,
        "white": 110,
        "frozen": 80,
        "soft": 35,
        "flash": 90,
        "caption_mismatch": 130,
        "caption_cropped": 130,
        "caption_glyphs": 130,
        "small_text": 8,
        "edge_text": 10,
        "caption_jump": 35,
        "unplanned_text": 85,
    }
    score = round(max(0, 100 - sum(weights[k] * v / n for k, v in counts.items())), 1)
    blockers = [
        k
        for k, v in counts.items()
        if (k.startswith("caption_") and v >= 1)
        or (k in ("black", "white", "flash") and v >= max(2, n * 0.02))
        or (k == "unplanned_text" and v >= max(2, n * 0.04))
        or (k == "frozen" and v / n > 0.18)
    ]
    return score, blockers


def audit(
    video: Path,
    folder: Path,
    mode="reference",
    cues=(),
    progress=lambda s: None,
    plates=None,
):
    folder.mkdir(parents=True, exist_ok=True)
    counts, words, shots = Counter(), Counter(), []
    prev = prev_delta = None
    still = 0
    next_thumb = 0.0
    thumb_count = 0
    previous_lines = {}
    with (
        av.open(str(video)) as container,
        (folder / "frames.jsonl").open("w", encoding="utf-8") as ledger,
    ):
        stream = container.streams.video[0]
        fps = float(stream.average_rate or 24)
        metadata = {
            "width": stream.width,
            "height": stream.height,
            "fps": round(fps, 3),
            "audio": bool(container.streams.audio),
            "codec": stream.codec_context.name,
        }
        n = 0
        for frame in container.decode(stream):
            pts = float(frame.time) if frame.time is not None else n / fps
            if n == 0:
                start = pts
            t = round(pts - start, 5)
            img = frame.to_ndarray(format="bgr24")
            gray = cv2.cvtColor(
                cv2.resize(img, (144, 160), interpolation=cv2.INTER_AREA),
                cv2.COLOR_BGR2GRAY,
            )
            brightness = float(gray.mean())
            sharp = float(cv2.Laplacian(gray, cv2.CV_32F).var())
            delta = float(cv2.absdiff(prev, gray).mean()) if prev is not None else 0.0
            change = float(gray.mean() - prev.mean()) if prev is not None else 0.0
            still = still + 1 if prev is not None and delta < 0.45 else 0
            flags = []
            if brightness < 8 and gray.std() < 10:
                flags.append("black")
            if brightness > 248 and gray.std() < 10:
                flags.append("white")
            if sharp < 8:
                flags.append("soft")
            if still >= max(3, round(0.45 * fps)):
                flags.append("frozen")
            if (
                prev_delta is not None
                and abs(change) > 38
                and change * prev_delta < -1200
            ):
                flags.append("flash")
            cut = n == 0 or (
                delta > 35 and (not shots or t - shots[-1]["start"] > 0.28)
            )
            if cut:
                shots.append({"start": t, "frame": n})
            lines = read_text(img)
            words.update(l["text"] for l in lines)
            caption = next(
                (cue["text"] for cue in cues if cue["start"] <= t < cue["end"]), ""
            )
            flags += text_checks(lines, caption, mode)
            glyph_score = None
            if caption and plates:
                cue_id = next(
                    i for i, c in enumerate(cues) if c["start"] <= t < c["end"]
                )
                glyph_score = round(glyph_quality(img, plates[cue_id]), 4)
                if glyph_score < 0.84:
                    flags.append("caption_glyphs")
            positions = {
                norm(l["text"]): l["box"]
                for l in lines
                if l["confidence"] > 0.75 and len(norm(l["text"])) >= 3
            }
            if mode == "reference":
                if any(b[3] - b[1] < 0.018 for b in positions.values()):
                    flags.append("small_text")
                if any(
                    b[0] < 0.035 or b[2] > 0.95 or b[1] < 0.035 or b[3] > 0.93
                    for b in positions.values()
                ):
                    flags.append("edge_text")
                if not cut and any(
                    k in previous_lines and abs(b[1] - previous_lines[k][1]) > 0.055
                    for k, b in positions.items()
                ):
                    flags.append("caption_jump")
            previous_lines = positions
            counts.update(flags)
            row = {
                "frame": n,
                "time": t,
                "expected": caption,
                "glyph_match": glyph_score,
                "ocr": lines,
                "luma": round(brightness, 1),
                "sharpness": round(sharp, 2),
                "motion": round(delta, 2),
                "flags": flags,
            }
            ledger.write(json.dumps(row, ensure_ascii=False) + "\n")
            if (cut or t >= next_thumb) and thumb_count < 60:
                thumb = cv2.resize(
                    img, None, fx=320 / max(img.shape[:2]), fy=320 / max(img.shape[:2])
                )
                cv2.imwrite(str(folder / f"key-{n:06}.jpg"), thumb)
                thumb_count += 1
                next_thumb = t + 1
            prev, prev_delta = gray, change
            n += 1
            if n == 1 or n % 12 == 0:
                ledger.flush()
                progress(f"OCR · {n} frames · {t:.1f}s")
    if not n:
        raise ValueError("Video has no decodable frames")
    duration = round(t + 1 / fps, 3)
    for i, shot in enumerate(shots):
        shot["end"] = shots[i + 1]["start"] if i + 1 < len(shots) else duration
    score, blockers = judge(counts, n)
    report = {
        **metadata,
        "frames": n,
        "duration": duration,
        "mode": mode,
        "score": score,
        "passed": score >= 75 and not blockers,
        "blockers": blockers,
        "flag_counts": dict(counts),
        "shots": shots,
        "frequent_text": words.most_common(20),
        "ledger": "frames.jsonl",
        "sound": audio_metrics(video, metadata["audio"]),
    }
    save(folder / "report.json", report)
    return report
