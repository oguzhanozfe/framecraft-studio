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


def synthesize(text, output, *, seconds=30, segments=None):
    """Keep the natural delivery. Reject overlength reads instead of speeding them up."""
    python = os.getenv("FRAMECRAFT_VOICE_PYTHON", sys.executable)
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
        "provider": "FreyaTTS",
        "seconds": round(length, 3),
        "tempo": 1.0,
        "text": text,
        "segments": timings,
    }


if __name__ == "__main__":
    params = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    if params["source"]:
        sys.path.insert(0, params["source"])
    from freyatts import FreyaTTS

    model = FreyaTTS.from_pretrained(params["weights"], device=params["device"])
    if params.get("auditions"):
        for n, line in enumerate(params["auditions"]):
            model.save_wav(
                model.synthesize(line, steps=32), str(Path(sys.argv[2]) / f"{n}.wav")
            )
        sys.exit(0)
    if params.get("segments"):
        import numpy as np

        rate = model.sample_rate
        takes = []
        for line in params["segments"]:
            take = model.synthesize(line, steps=32)
            audible = np.flatnonzero(np.abs(take) > 0.004)
            if audible.size < rate // 20:
                raise ValueError("Silent Freya read")
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
        sample = model.synthesize(params["text"], steps=32)
    model.save_wav(sample, sys.argv[2])
