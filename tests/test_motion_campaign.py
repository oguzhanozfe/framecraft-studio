import subprocess
import numpy as np
import pytest
from framecraft.campaign_motion import render, repeated_motion, spans


def test_voice_alignment_catches_a_delayed_right_channel_and_echo():
    from framecraft.voice_alignment import RATE, compare

    rng = np.random.default_rng(51)
    voice = rng.normal(0, 0.12, RATE * 3).astype("float32")
    noise = rng.normal(0, 0.012, (len(voice), 2)).astype("float32")
    mix = np.column_stack([voice, voice]) + noise
    assert compare(voice, mix)["passed"]
    delayed = np.roll(voice, RATE // 10)
    assert not compare(voice, np.column_stack([voice, delayed]))["passed"]
    assert not compare(voice, mix + 0.65 * delayed[:, None])["passed"]


def test_phone_pan_preserves_scene_structure():
    import cv2
    from framecraft.campaign_motion import continuous_pan

    rng = np.random.default_rng(91)
    scene = cv2.GaussianBlur(
        rng.integers(0, 256, (180, 160), dtype=np.uint8), (3, 3), 0
    )
    moved = cv2.warpAffine(scene, np.float32([[1, 0, 8], [0, 1, -2]]), (160, 180))
    assert continuous_pan(scene, moved)
    assert not continuous_pan(scene, np.flipud(moved).copy())


def test_camera_roll_reshoots_only_the_marked_interior(tmp_path, monkeypatch):
    import cv2
    from framecraft import phone_review
    from framecraft.core import read

    original = tmp_path / "take.mp4"

    def make_clip(path, frames):
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24, (96, 160)
        )
        for f in range(frames):
            writer.write(np.full((160, 96, 3), 25 + f % 174, np.uint8))
        writer.release()

    make_clip(original, 120)
    calls = []

    def guided(provider, prompt, seed, shape, target, progress, **opts):
        calls.append(opts)
        make_clip(target, opts["frames"] + 5)

    monkeypatch.setattr(phone_review, "generate", guided)
    monkeypatch.setattr(phone_review, "inspect", lambda *args: {"passed": True})
    report = {
        "signature": "fixture",
        "sideways_tiles": [5],
        "impossible_anatomy_tiles": [],
    }
    fixed = phone_review.patch_roll(original, report, "A phone on a desk", None)
    assert calls[0]["reference"].is_file() and calls[0]["end_reference"].is_file()
    left, right = read(fixed.with_suffix(".json"))["replaced_frames"]
    assert 0 < left < right < 119
    cap = cv2.VideoCapture(str(fixed))
    try:
        assert round(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 120
    finally:
        cap.release()
    assert phone_review.patch_roll(original, report, "A phone on a desk", None) == fixed
    assert len(calls) == 1


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
