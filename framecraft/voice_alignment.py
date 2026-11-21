"""Locate the dry narrator independently in each channel of the encoded master."""

import subprocess

import cv2
import numpy as np


RATE = 8000


def decode(path, channels):
    data = subprocess.check_output(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-ar",
            str(RATE),
            "-ac",
            str(channels),
            "-f",
            "f32le",
            "pipe:1",
        ]
    )
    return np.frombuffer(data, dtype="<f4").reshape(-1, channels)


def compare(voice, mix):
    block, margin = RATE // 2, RATE // 4
    n = min(len(voice), len(mix))
    probes = []
    for group in np.array_split(np.arange(margin, n - block - margin, block // 2), 5):
        if not len(group):
            continue
        start = max(group, key=lambda t: np.mean(voice[t : t + block] ** 2))
        source = np.ascontiguousarray(voice[start : start + block].reshape(-1, 1))
        if np.mean(source**2) < 0.00004:
            continue
        values = []
        for channel in mix.T:
            window = np.ascontiguousarray(
                channel[start - margin : start + block + margin].reshape(-1, 1)
            )
            correlation = cv2.matchTemplate(
                window, source, cv2.TM_CCOEFF_NORMED
            ).ravel()
            pick = int(np.argmax(correlation))
            signal = window[pick : pick + block, 0]
            dry = source[:, 0]
            gain = np.dot(signal, dry) / max(float(np.dot(dry, dry)), 1e-9)
            residual = signal - gain * dry
            # Fit an independent delayed copy over the interior of this window.
            margin2 = RATE // 5
            extended = np.ascontiguousarray(voice[start - margin2 : start + block])
            shifted_fit = cv2.matchTemplate(
                extended[:, None], residual[:, None], cv2.TM_CCORR
            ).ravel()
            tail = shifted_fit[: margin2 - RATE // 40]
            echo = max(0, float(tail.max())) / max(float(np.dot(dry, dry) * gain), 1e-8)
            values.append(
                {
                    "delay_ms": round((pick - margin) * 1000 / RATE, 1),
                    "correlation": round(float(correlation[pick]), 3),
                    "secondary_voice_ratio": round(echo, 3),
                }
            )
        probes.append({"at": round(start / RATE, 3), "channels": values})
    passed = bool(probes) and all(
        abs(c["delay_ms"]) <= 35
        and c["correlation"] >= 0.58
        and c["secondary_voice_ratio"] < 0.32
        for p in probes
        for c in p["channels"]
    )
    return {"passed": passed, "probes": probes}


def measure(voice, master):
    return compare(decode(voice, 1)[:, 0], decode(master, 2))
