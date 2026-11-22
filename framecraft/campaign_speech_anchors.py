"""Local Turkish acoustic landmarks for bilabial closures and vowel openings.

Pass the actual spoken text (expand numbers and abbreviations) to force-align
the dialogue. A forced character with low acoustic posterior is retained for
inspection and never presented as an independently observed visual target.
The tested ONNX model is onnx-community/wav2vec2-large-xlsr-turkish-ONNX.
Supply its config.json, vocab.json, and quantized model renamed to model.onnx.
"""

import itertools
import json
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np


@lru_cache(maxsize=2)
def _ctc_model(folder):
    import onnxruntime as ort

    config = json.loads((folder / "config.json").read_text(encoding="utf8"))
    vocab = json.loads((folder / "vocab.json").read_text(encoding="utf8"))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 3
    model = ort.InferenceSession(
        str(folder / "model.onnx"), options, providers=["CPUExecutionProvider"]
    )
    stride = 1
    receptive = 1
    for kernel, hop in zip(config["conv_kernel"], config["conv_stride"], strict=True):
        receptive += (kernel - 1) * stride
        stride *= hop
    return model, vocab, stride / 16000, (receptive - 1) / 32000


def _forced_tokens(logp, target, blank):
    """Viterbi on the CTC alignment graph (repeated letters require a blank)."""
    states = np.full(len(target) * 2 + 1, blank, dtype=int)
    states[1::2] = target
    transition = np.zeros(len(states), dtype=bool)
    transition[2:] = (states[2:] != blank) & (states[2:] != states[:-2])
    prev = np.full(len(states), -np.inf)
    prev[0] = 0
    back = np.zeros((len(logp), len(states)), dtype=np.uint8)
    for t, probs in enumerate(logp):
        one = np.r_[-np.inf, prev[:-1]]
        two = np.where(transition, np.r_[-np.inf, -np.inf, prev[:-2]], -np.inf)
        moves = np.stack([prev, one, two])
        back[t] = moves.argmax(axis=0)
        prev = moves.max(axis=0) + probs[states]
    pos = len(states) - 1 if prev[-1] >= prev[-2] else len(states) - 2
    if not np.isfinite(prev[pos]):
        raise ValueError("Speech is too short for the provided transcript")
    path = np.empty(len(logp), dtype=int)
    for t in range(len(logp) - 1, -1, -1):
        path[t] = pos
        pos -= int(back[t, pos])
    return [
        (int(letter), np.flatnonzero(path == 2 * i + 1))
        for i, letter in enumerate(target)
    ]


def acoustic_anchors(audio, model_dir, *, start=0.0, seconds=None, transcript=None):
    """Return acoustic tokens and confident viseme timestamps relative to this slice."""
    if start < 0 or (seconds is not None and seconds <= 0):
        raise ValueError("Invalid acoustic interval")
    args = ["ffmpeg", "-nostdin", "-v", "error", "-ss", str(start), "-i", str(audio)]
    if seconds is not None:
        args += ["-t", str(seconds)]
    args += ["-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "-"]
    waveform = np.frombuffer(subprocess.check_output(args), dtype="<f4")
    if len(waveform) < 1600 or np.max(np.abs(waveform)) < 0.001:
        raise ValueError("Insufficient audible dialogue for anchor alignment")
    model, vocab, hop, center = _ctc_model(Path(model_dir))
    x = (waveform - waveform.mean()) / np.sqrt(waveform.var() + 1e-7)
    feed = {model.get_inputs()[0].name: x[None]}
    if len(model.get_inputs()) > 1:
        feed[model.get_inputs()[1].name] = np.ones_like(x[None], dtype=np.int64)
    logits = model.run(None, feed)[0][0]
    shifted = logits - logits.max(axis=1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    symbols = {value: symbol for symbol, value in vocab.items()}
    # Model conversions vary in their exported <s>/<pad> blank id; combine
    # acoustically empty symbols rather than confusing a blank with a letter.
    empty = [v for k, v in vocab.items() if k in {"<pad>", "<s>", "</s>", "[PAD]"}]
    blank = empty[0]
    merged = np.logaddexp.reduce(logp[:, empty], axis=1)
    logp[:, empty] = -np.inf
    logp[:, blank] = merged
    if transcript is not None:
        folded = transcript.replace("İ", "i").replace("I", "ı").lower()
        if any(c.isdigit() for c in folded):
            raise ValueError("Pass the spoken expansion for digits in the script")
        words = [
            "".join(k for k in word if k in vocab and len(k) == 1)
            for word in folded.split()
        ]
        normalized = "|".join(w for w in words if w)
        if not normalized:
            raise ValueError("No spoken letters found in transcript")
        align = _forced_tokens(logp, [vocab[c] for c in normalized], blank)
    else:
        align = []
        for label, group in itertools.groupby(
            enumerate(logp.argmax(axis=1)), key=lambda x: x[1]
        ):
            indices = np.array([t for t, _ in group])
            if label != blank:
                align.append((int(label), indices))
    tokens, anchors = [], []
    for label, frames in align:
        letter = symbols[label]
        if len(letter) != 1:
            continue
        peak = frames[np.argmax(logp[frames, label])]
        confidence = float(np.exp(logp[peak, label]))
        token = {
            "letter": letter,
            "time": round(center + peak * hop, 4),
            "start": round(max(0.0, center + (frames[0] - 0.5) * hop), 4),
            "end": round(center + (frames[-1] + 0.5) * hop, 4),
            "confidence": round(confidence, 4),
        }
        tokens.append(token)
        if letter in "pbmaâeoöuüiî" and confidence >= 0.55:
            if letter in "pbm":
                kind = "bilabial"
            elif letter in "aâe":
                kind = "open_vowel"
            elif letter in "oöuü":
                kind = "rounded_vowel"
            else:
                kind = "spread_vowel"
            anchors.append({**token, "type": kind})
    return {
        "mode": "forced" if transcript is not None else "acoustic",
        "tokens": tokens,
        "anchors": anchors,
    }
