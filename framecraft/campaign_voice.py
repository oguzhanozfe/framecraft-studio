"""Synthesize a continuous Turkish read using an isolated local FreyaTTS runtime."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import wave


def duration(path):
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def place_read(voice, target, segments, beats):
    """Place complete, natural-speed phrases against picture without duplicating a stem."""
    import numpy as np

    if len(segments) != len(beats) or not beats or min(beats) <= 0:
        raise ValueError("One narration phrase per camera beat is required")
    with wave.open(str(voice), "rb") as source:
        if source.getsampwidth() != 2 or source.getnchannels() != 1:
            raise ValueError("Expected a mono 16-bit dry narration stem")
        rate = source.getframerate()
        raw = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")
    result = np.zeros(round(sum(beats) * rate), dtype="<i2")
    timings = []
    cursor = 0
    for i, (segment, length) in enumerate(zip(segments, beats, strict=True)):
        phrase = raw[round(segment["start"] * rate) : round(segment["end"] * rate)]
        # The next thought can lead the corresponding camera movement by a quarter second.
        start = max(cursor + round(rate * 0.15), round(rate * (sum(beats[:i]) - 0.25)))
        end = start + len(phrase)
        if (
            not len(phrase)
            or end > len(result)
            or end > (sum(beats[: i + 1]) + 0.45) * rate
        ):
            raise ValueError(f"Phrase {i + 1} does not fit its camera beat")
        result[start:end] = phrase
        cursor = end
        timings.append(
            {
                "start": round(start / rate, 3),
                "end": round(end / rate, 3),
                "text": segment["text"],
            }
        )
    with wave.open(str(target), "wb") as dst:
        dst.setparams((1, 2, rate, 0, "NONE", "uncompressed"))
        dst.writeframes(result.tobytes())
    return timings


def synthesize(text, output, *, seconds=30, segments=None, profile="freya"):
    """Keep the natural delivery. Reject overlength reads instead of speeding them up."""
    python = os.getenv("FRAMECRAFT_VOICE_PYTHON", sys.executable)
    if profile not in {"freya", "pocket/female_1", "pocket/female_2"}:
        raise ValueError("Unknown narration profile")
    if profile.startswith("pocket/"):
        python = os.getenv("FRAMECRAFT_POCKET_PYTHON") or sys.executable
    source = os.getenv("FRAMECRAFT_FREYA_SOURCE")
    weights = os.getenv("FRAMECRAFT_FREYA_MODEL", "freyavoice/Freya-TTS")
    device = os.getenv("FRAMECRAFT_VOICE_DEVICE", "cuda")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as work:
        work = Path(work)
        request = work / "request.json"
        wav = work / "read.wav"
        request.write_text(
            json.dumps(
                dict(
                    text=text,
                    weights=weights,
                    device=device,
                    source=source,
                    segments=segments,
                    seconds=seconds,
                    profile=profile,
                )
            ),
            encoding="utf-8",
        )
        completed = subprocess.run(
            [python, str(Path(__file__).resolve()), str(request), str(wav)]
        )
        if completed.returncode == 4 and (work / "rejected.json").is_file():
            raise ValueError(json.loads((work / "rejected.json").read_text())["reason"])
        completed.check_returncode()
        length = duration(wav)
        if length > seconds - 0.12 or length < 1:
            raise ValueError(
                f"Natural read lasts {length:.2f}s for a {seconds}s cut; revise copy."
            )
        timings = json.loads((work / "timings.json").read_text()) if segments else []
        wav.replace(output)
    return {
        "provider": "Pocket TTS Turkish"
        if profile.startswith("pocket/")
        else "FreyaTTS",
        "profile": profile,
        "seconds": round(length, 3),
        "tempo": 1.0,
        "text": text,
        "segments": timings,
    }


if __name__ == "__main__":
    params = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    profile = params.get("profile", "freya")
    if profile.startswith("pocket/"):
        from pocket_tts_turkish import TurkishTTS, save_wav

        model = TurkishTTS.from_pretrained(device="cpu")
        voice = profile.split("/")[1]
        if voice not in model.voices:
            raise ValueError("Selected voice is unavailable")

        def speak(line):
            return model.generate(line, voice=voice, sentence_gap_ms=220, seed=737)

        def write_audio(data, file):
            save_wav(file, data, model.sample_rate)
    else:
        if params["source"]:
            sys.path.insert(0, params["source"])
        from freyatts import FreyaTTS

        model = FreyaTTS.from_pretrained(params["weights"], device=params["device"])

        def speak(line):
            return model.synthesize(line, steps=32)

        write_audio = model.save_wav
    if params.get("auditions"):
        for n, line in enumerate(params["auditions"]):
            write_audio(speak(line), str(Path(sys.argv[2]) / f"{n}.wav"))
        sys.exit(0)
    if params.get("segments"):
        import numpy as np

        rate = model.sample_rate
        takes = []
        for line in params["segments"]:
            take = speak(line)
            audible = np.flatnonzero(np.abs(take) > 0.004)
            if audible.size < rate // 20:
                raise ValueError("Silent narration take")
            takes.append(
                take[max(0, audible[0] - rate // 30) : audible[-1] + rate // 12]
            )
        total = sum(len(t) / rate for t in takes)
        room = params["seconds"] - 0.7 - total
        if room < 0.14 * (len(takes) - 1):
            reason = f"Natural narration needs a shorter line ({total:.1f}s)"
            (Path(sys.argv[2]).parent / "rejected.json").write_text(
                json.dumps({"reason": reason})
            )
            sys.exit(4)
        gap = round(rate * min(1.05, room / max(1, len(takes) - 1)))
        ending = round(rate * max(0, room - gap * (len(takes) - 1) / rate))
        pieces = []
        cursor = 0
        timings = []
        for i, (line, take) in enumerate(zip(params["segments"], takes)):
            # Give the last visual reveal breathing room before the final opinion.
            if i:
                pause = gap + (
                    min(ending, round(rate * 1.4)) if i == len(takes) - 1 else 0
                )
                pieces.append(np.zeros(pause, dtype=np.float32))
                cursor += pause
            timings.append(
                {
                    "start": round(cursor / rate, 3),
                    "end": round((cursor + len(take)) / rate, 3),
                    "text": line,
                }
            )
            pieces.append(take)
            cursor += len(take)
        sample = np.concatenate(pieces)
        (Path(sys.argv[2]).parent / "timings.json").write_text(json.dumps(timings))
    else:
        sample = speak(params["text"])
    write_audio(sample, sys.argv[2])
