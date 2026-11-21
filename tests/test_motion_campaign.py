import subprocess
import numpy as np
import pytest
from framecraft.campaign_motion import render, repeated_motion, spans


def test_reframe_windows_keep_even_video_coordinates_inside_the_source():
    from framecraft.motion_reframe import windows

    options = windows(449, 801)
    assert len(options) == len(set(options)) == 27
    for x, y, w, h in options:
        assert 0 <= x < x + w <= 449 and 0 <= y < y + h <= 801
        assert all(not v % 2 for v in (x, y, w, h))


def test_execution_timeout_starts_when_the_render_leaves_the_queue(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace
    import httpx
    from framecraft import providers

    clock = [0]
    ticks = iter([100, 3550, 3650, 3700])

    def respond(request):
        path = request.url.path
        if path == "/prompt":
            return httpx.Response(200, json={"prompt_id": "job"})
        if path == "/queue":
            pending = clock[0] < 3600
            return httpx.Response(
                200,
                json={
                    "queue_pending": [[1, "job"]] if pending else [],
                    "queue_running": [] if pending else [[1, "job"]],
                },
            )
        if path == "/history/job":
            record = {
                "job": {
                    "outputs": {
                        "15": {
                            "images": [
                                {
                                    "filename": "cut.mp4",
                                    "subfolder": "video",
                                    "type": "output",
                                }
                            ]
                        }
                    }
                }
            }
            return httpx.Response(200, json=record if clock[0] == 3700 else {})
        if path == "/view":
            return httpx.Response(200, content=b"rendered")
        raise AssertionError(path)

    client = httpx.Client(
        base_url="http://localhost", transport=httpx.MockTransport(respond)
    )
    monitor = SimpleNamespace(
        active=False,
        close=lambda: None,
        inspect=lambda *args: clock.__setitem__(0, next(ticks)),
    )
    monkeypatch.setattr(providers, "PreviewMonitor", lambda *args: monitor)
    monkeypatch.setattr(providers, "httpx", SimpleNamespace(Client=lambda **kw: client))
    monkeypatch.setattr(providers, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    out = tmp_path / "cut.mp4"
    assert (
        providers.generate("h3", "demo", 5, "vertical", out, lambda text: None) == out
    )
    assert out.read_bytes() == b"rendered"


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


def test_approved_excerpt_rejects_source_changes_and_retiming(tmp_path):
    from framecraft.campaign_remix import approved_shot
    from framecraft.core import save
    import hashlib

    shot = tmp_path / "raw.mp4"
    shot.write_bytes(b"original take")
    master = tmp_path / "old.mp4"
    master.write_bytes(b"reviewed edit")
    spec = tmp_path / "motion.json"
    save(spec, {"output": master.name})
    save(
        tmp_path / "motion-assessment.json",
        {
            "passed": True,
            "sha256": hashlib.sha256(master.read_bytes()).hexdigest(),
            "timeline": [
                {"source": shot.name, "from": 0, "to": 60, "source_start_seconds": 1.0}
            ],
        },
    )
    with pytest.raises(ValueError, match="within"):
        approved_shot(spec, 0, 1.8, 0.5, tmp_path)
    locked = approved_shot(spec, 0, 0.5, 1.0, tmp_path)
    with pytest.raises(ValueError, match="interval"):
        spans({"shots": [{**locked, "start": 1.6}]}, tmp_path)
    shot.write_bytes(b"regenerated take")
    with pytest.raises(ValueError, match="footage has changed"):
        spans({"shots": [locked]}, tmp_path)


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
    # A music-only recut at the end of a physical take must preserve the exact
    # duration and dissolve before the incoming caption is composited.
    shots[1]["join"] = "dissolve"
    shots[1].update(layout="caption", headline="Cyberpunk 2077")
    render(
        {
            "preview_seconds": 2,
            "shots": shots,
            "audio_mode": "music",
            "score": "voice.wav",
        },
        tmp_path,
    )
    result = json.loads((tmp_path / "motion-assessment.json").read_text())
    assert result["passed"] and result["frames"] == 60
    assert result["joins"][0]["join"] == "dissolve"
    assert result["voice"] is None and result["score_sha256"]
