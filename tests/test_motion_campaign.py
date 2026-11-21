import subprocess
import numpy as np
import pytest
from framecraft.campaign_motion import render, repeated_motion, spans


def test_short_source_and_reused_interval_fail_before_encoding(tmp_path):
    src = tmp_path / "scene.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x64:rate=30:duration=2",
            "-c:v",
            "libx264",
            "-y",
            str(src),
        ],
        check=True,
    )
    with pytest.raises(ValueError, match="too short"):
        spans({"shots": [{"file": src.name, "seconds": 3}]}, tmp_path)
    with pytest.raises(ValueError, match="Repeated footage"):
        spans(
            {
                "shots": [
                    {"file": src.name, "seconds": 1.5},
                    {"file": src.name, "start": 1, "seconds": 1},
                ]
            },
            tmp_path,
        )
    with pytest.raises(ValueError, match="exactly 30s"):
        spans(
            {
                "shots": [
                    {"file": src.name, "seconds": 1},
                    {"file": src.name, "start": 1, "seconds": 1},
                ]
            },
            tmp_path,
        )
    with pytest.raises(ValueError, match="Motion footage required"):
        spans({"shots": [{"file": "product.jpg", "seconds": 30}]}, tmp_path)


def test_repeat_detection_survives_codec_noise():
    rng = np.random.default_rng(13)
    original = rng.integers(8, 241, (30, 13, 16), dtype=np.uint8)
    assert not repeated_motion(original)
    reused = np.clip(original[6:21].astype(np.int16) + 1, 0, 255).astype(np.uint8)
    assert repeated_motion(np.concatenate([original, reused]))


def test_presenter_and_generated_take_with_different_color_metadata(tmp_path):
    import json

    shots = []
    for i, tags in enumerate(
        [[], ["-colorspace", "bt709", "-color_primaries", "bt709"]]
    ):
        name = f"take-{i}.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=144x256:rate=24:duration=1",
                "-vf",
                f"hue=h={i * 120}",
                *tags,
                "-c:v",
                "libx264",
                str(tmp_path / name),
            ],
            check=True,
        )
        shots.append({"file": name, "seconds": 1, "layout": "clean"})
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=225:duration=2",
            str(tmp_path / "voice.wav"),
        ],
        check=True,
    )
    output = render(
        {"preview_seconds": 2, "shots": shots, "voice": "voice.wav"}, tmp_path
    )
    result = json.loads((tmp_path / "motion-assessment.json").read_text())
    assert output.is_file() and result["passed"] and result["frames"] == 60
