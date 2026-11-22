import json
import shutil

import cv2
import numpy as np
import pytest

from framecraft import campaign_listening as listening


def test_quiet_cutaway_uses_own_resting_frame_and_preserves_motion(
    tmp_path, monkeypatch
):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg is required for video encoding")
    source, output = tmp_path / "source.avi", tmp_path / "quiet.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 64))
    assert writer.isOpened()
    for index in range(8):
        frame = np.full((64, 64, 3), 30, dtype=np.uint8)
        frame[:16] = 40 + 20 * index
        frame[32:48, 24:40] = 35 if index == 2 else 190
        writer.write(frame)
    writer.release()

    monkeypatch.setattr(listening, "track", lambda src, start, n: np.zeros((n, 4, 2)))
    monkeypatch.setattr(
        listening, "resting_references", lambda m: np.ones(len(m), dtype=int)
    )
    monkeypatch.setattr(
        listening, "face_boxes", lambda m, w, h: [[24, 32, 40, 48]] * len(m)
    )
    monkeypatch.setattr(listening, "lip_mask", lambda face, box: np.ones((16, 16, 1)))
    monkeypatch.setattr(
        listening,
        "rest_crop",
        lambda frames, mesh, i, ref, box: frames[ref][32:48, 24:40],
    )
    report = listening.render(source, output, start=0.1, seconds=0.5)
    assert report == json.loads(output.with_suffix(".source.json").read_text())
    assert report["frames"] == 5 and report["same_actor_resting_references"] == 1
    cap = cv2.VideoCapture(str(output))
    try:
        for i in range(5):
            ok, frame = cap.read()
            assert ok and frame[36:44, 30:36].mean() < 50
            assert abs(frame[4:10, 4:10].mean() - (60 + 20 * i)) < 8
        assert not cap.read()[0]
    finally:
        cap.release()

    old_hash = report["sha256"]
    monkeypatch.setattr(listening, "resting_references", lambda m: np.full(len(m), -1))
    with pytest.raises(ValueError, match="resting expression"):
        listening.render(source, output)
    assert listening.fingerprint(output) == old_hash
    with pytest.raises(ValueError, match="separate output"):
        listening.render(source, source)
