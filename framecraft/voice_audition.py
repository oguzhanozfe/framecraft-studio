"""Audition alternate factual readings of difficult Turkish names and numbers."""

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import wave
import numpy as np
from faster_whisper import WhisperModel
from num2words import num2words
from .audit import similarity
from .core import read, save


def canonical(text, pronunciation):
    for src, dst in pronunciation.items():
        text = re.sub(re.escape(src), dst, text, flags=re.IGNORECASE)
    return re.sub(r"\d+", lambda m: num2words(int(m.group()), lang="tr"), text)


def produce(brief, folder, asr):
    from .motion_copy import unsupported_testimony

    pronunciations = brief.get("pronunciation", {})
    banks = [
        [line for line in bank if not unsupported_testimony(line)]
        for bank in brief["spoken_beats"]
    ]
    if not all(banks):
        raise ValueError("No grounded narration left for a spoken beat")
    alternatives = [canonical(line, pronunciations) for bank in banks for line in bank]
    weights = os.getenv("FRAMECRAFT_FREYA_MODEL", "freyavoice/Freya-TTS")
    python = os.getenv("FRAMECRAFT_VOICE_PYTHON", sys.executable)
    with tempfile.TemporaryDirectory(dir=folder) as tmp:
        work = Path(tmp)
        request = work / "request.json"
        save(
            request,
            dict(
                auditions=alternatives,
                weights=weights,
                source=os.getenv("FRAMECRAFT_FREYA_SOURCE"),
                device=os.getenv("FRAMECRAFT_VOICE_DEVICE", "cuda"),
            ),
        )
        subprocess.run(
            [
                python,
                str(Path(__file__).with_name("campaign_voice.py")),
                str(request),
                str(work),
            ],
            check=True,
        )
        model = WhisperModel(str(asr), device="cpu", compute_type="int8", cpu_threads=4)
        takes = []
        choices = []
        idx = 0
        for bank in banks:
            options = []
            for original in bank:
                path = work / f"{idx}.wav"
                audio = subprocess.check_output(
                    [
                        "ffmpeg",
                        "-loglevel",
                        "error",
                        "-i",
                        str(path),
                        "-ar",
                        "16000",
                        "-ac",
                        "1",
                        "-f",
                        "f32le",
                        "pipe:1",
                    ]
                )
                parts, _ = model.transcribe(
                    np.frombuffer(audio, dtype="<f4"), language="tr", beam_size=5
                )
                heard = " ".join(part.text.strip() for part in parts)
                expected = alternatives[idx]
                score = similarity(canonical(heard, pronunciations), expected)
                options.append(
                    dict(
                        index=idx,
                        original=original,
                        spoken=expected,
                        heard=heard,
                        score=round(score, 3),
                    )
                )
                idx += 1
            winner = max(options, key=lambda o: o["score"])
            with wave.open(str(work / f"{winner['index']}.wav"), "rb") as h:
                rate = h.getframerate()
                samples = np.frombuffer(h.readframes(h.getnframes()), dtype="<i2")
            active = np.flatnonzero(abs(samples.astype(np.int32)) > 110)
            if not len(active):
                raise ValueError("Empty take")
            takes.append(
                samples[max(0, active[0] - rate // 25) : active[-1] + rate // 10]
            )
            choices.append({"selected": winner, "options": options})
        duration = sum(len(t) for t in takes) / rate
        gap = (29.35 - duration) / (len(banks) - 1)
        if gap < 0.13:
            raise ValueError("Selected reads exceed the thirty-second timeline")
        assembled = []
        cursor = 0
        segments = []
        for i, take in enumerate(takes):
            if i:
                pause = round(rate * gap)
                assembled.append(np.zeros(pause, dtype="<i2"))
                cursor += pause
            segments.append(
                {
                    "start": round(cursor / rate, 3),
                    "end": round((cursor + len(take)) / rate, 3),
                    "text": choices[i]["selected"]["spoken"],
                }
            )
            assembled.append(take)
            cursor += len(take)
        with wave.open(str(work / "final.wav"), "wb") as result:
            result.setparams((1, 2, rate, 0, "NONE", "uncompressed"))
            result.writeframes(np.concatenate(assembled).tobytes())
        if min(c["selected"]["score"] for c in choices) < 0.69:
            save(folder / "audition-rejected.json", choices)
            raise ValueError(
                "No sufficiently intelligible reading for one or more beats"
            )
        (work / "final.wav").replace(folder / "delivery-voice.wav")
        save(
            folder / "delivery-voice.json",
            dict(
                text=" ".join(s["text"] for s in segments),
                segments=segments,
                seconds=round(cursor / rate, 3),
                tempo=1.0,
                provider="FreyaTTS",
                auditions=choices,
            ),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("brief", type=Path)
    parser.add_argument("asr", type=Path)
    args = parser.parse_args()
    produce(read(args.brief), args.brief.parent, args.asr)
