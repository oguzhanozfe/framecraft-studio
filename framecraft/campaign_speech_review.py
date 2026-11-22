"""Run physical speech checks on the delivered face using the optional CPU runtime."""

import json
import os
import subprocess
import tempfile
from pathlib import Path

from .campaign_face import landmark_runtime


def review(
    video,
    voice,
    *,
    start=0.0,
    seconds=None,
    audio_start=0.0,
    transcript=None,
    anchors=None,
):
    python, landmark_model = landmark_runtime()
    home = Path(os.environ.get("FRAMECRAFT_HOME", Path.home() / ".framecraft"))
    acoustics = Path(
        os.environ.get(
            "FRAMECRAFT_SPEECH_ANCHORS", home / "models/turkish-speech-anchors-large"
        )
    )
    if not landmark_model.is_file() or (anchors is None and not acoustics.is_dir()):
        raise FileNotFoundError(
            "Facial speech review requires the Face Landmarker task and the Turkish acoustic model; configure FRAMECRAFT_FACE_MODEL and FRAMECRAFT_SPEECH_ANCHORS."
        )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(Path(__file__).resolve().parents[1]), env.get("PYTHONPATH", "")]
    )
    with tempfile.TemporaryDirectory() as temp:
        folder = Path(temp)
        report = folder / "speech.json"
        cmd = [
            python,
            "-m",
            "framecraft.campaign_articulation",
            str(video),
            str(voice),
            str(landmark_model),
            "--start",
            str(start),
            "--audio-start",
            str(audio_start),
            "--report",
            str(report),
        ]
        if seconds is not None:
            cmd.extend(["--seconds", str(seconds)])
        if anchors is not None:
            path = folder / "anchors.json"
            path.write_text(json.dumps(anchors), encoding="utf8")
            cmd.extend(["--anchors", str(path)])
        else:
            cmd.extend(["--acoustic-model", str(acoustics)])
            if transcript and not any(c.isdigit() for c in transcript):
                cmd.extend(["--transcript", transcript])
        proc = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            check=False,
            text=True,
            encoding="utf8",
            errors="replace",
        )
        if proc.returncode not in (0, 2) or not report.is_file():
            raise RuntimeError(
                "Could not measure the delivered mouth: " + proc.stderr[-1400:]
            )
        return json.loads(report.read_text(encoding="utf8"))
