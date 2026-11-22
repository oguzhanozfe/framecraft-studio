import json
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from framecraft.campaign_latentsync import (
    calibration,
    frame_count,
    reference_indices,
    render_registered_take,
    rest_seam,
    speaking_frames,
)


def test_listening_transition_requires_both_closed_lips_and_matching_pose():
    quiet = np.zeros((5, 478, 2))
    quiet[:, 263, 0] = 100
    moving = quiet.copy()
    moving[:2, 14, 1] = 12
    moving[3:] += 10
    index, report = rest_seam(moving, quiet)
    assert index == 2 and report["opening"] == 0
    with pytest.raises(ValueError, match="pose-matched"):
        rest_seam(moving + 20, quiet)


def test_reference_uses_independent_identity_and_closed_rest():
    gap = np.array([0.01, 0.3, 0.21, 0.007, 0.19, 0.2])
    active = np.array([False, True, True, False, False, True])
    chosen = reference_indices(gap, active, 6)
    assert list(chosen[active]) == [4, 5, 2]
    assert np.all(gap[chosen[~active]] < 0.04)
    with pytest.raises(ValueError):
        reference_indices(np.ones(6) * 0.25, active, 6)
    with pytest.raises(ValueError):
        reference_indices(gap, np.ones(7, bool), 6)


def test_does_not_loop_short_source_or_truncate_fps():
    assert frame_count(5.9, 25) == 148
    assert frame_count(5.9, 30) == 177
    with pytest.raises(ValueError):
        frame_count(0, 25)


def test_speech_rest_uses_real_waveform(tmp_path):
    source = np.zeros(24000, dtype="<i2")
    source[4800:9600] = 5000
    p = tmp_path / "voice.wav"
    with wave.open(str(p), "wb") as stream:
        stream.setparams((1, 2, 24000, 0, "NONE", "NONE"))
        stream.writeframes(source.tobytes())
    active = speaking_frames(p, 25)
    assert active[5:10].all() and not active[15:].any() and not active[:2].any()


def test_calibration_uses_entire_outline():
    anchors = np.array([[[10.0, 10.0], [40.0, 10.0], [25.0, 20.0]]] * 4)
    lips = np.array([[[20.0, 40.0], [30.0, 41.0], [25.0, 43.0]]] * 4)
    target = anchors[0]
    shift, covered = calibration(anchors, lips, target, (70, 80), np.zeros((80, 70)))
    assert covered == 1.0 and np.isfinite(shift).all()
    with pytest.raises(ValueError):
        calibration(anchors, lips, target, (70, 80), np.ones((80, 70)))


def test_registered_take_preserves_original_onset_and_reuses_source(
    tmp_path, monkeypatch
):
    from framecraft import campaign_face, campaign_listening
    from framecraft.campaign_cast import digest

    source, voice, task = [
        tmp_path / name for name in ("take.mp4", "reading.wav", "face.task")
    ]
    source.write_bytes(b"source")
    task.write_bytes(b"landmark task")
    samples = np.full(6000, 4500, dtype="<i2").tobytes()
    with wave.open(str(voice), "wb") as out:
        out.setparams((1, 2, 24000, 0, "NONE", "NONE"))
        out.writeframes(samples)
    monkeypatch.setattr(campaign_face, "landmark_runtime", lambda: ("landmarks", task))

    def no_rest(*args, **kwargs):
        raise ValueError("No matching neutral pose")

    monkeypatch.setattr(campaign_listening, "render", no_rest)
    calls = []

    def invoke(args, **kwargs):
        if args[0] == "ffmpeg":
            assert args[args.index("-ss") + 1] == "0.5"
            Path(args[-1]).write_bytes(b"cut")
        elif args[0] == "landmarks":
            Path(args[-1]).write_bytes(b"tracked")
        else:
            with wave.open(args[8], "rb") as bed:
                data = bed.readframes(bed.getnframes())
            assert data == bytes(9600) + samples + bytes(96000 - 9600 - len(samples))
            rendered = Path(args[10])
            rendered.write_bytes(b"model result")
            return subprocess.CompletedProcess(
                args, 0, stdout=json.dumps({"video_sha256": digest(rendered)})
            )
        calls.append(args[0])
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(subprocess, "run", invoke)
    options = {
        "source_start": 0.5,
        "seconds": 2.0,
        "voice_onset": 0.2,
        "repo": tmp_path,
        "checkpoints": tmp_path,
        "vae": tmp_path,
        "work_dir": tmp_path / "cache",
        "model_python": "diffusion",
    }
    for n in range(2):
        output = tmp_path / f"result{n}.mp4"
        report = render_registered_take(source, voice, output, **options)
        assert report["video_sha256"] == digest(output)
        assert report["source_sha256"] == digest(source)
        assert json.loads(output.with_suffix(".json").read_text())["voice_onset"] == 0.2
    assert calls == ["ffmpeg", "landmarks"]
