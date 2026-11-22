import cv2
import numpy as np

from framecraft import campaign_face_integrity as integrity
from framecraft.campaign_face import LOWER_FACE


def reference_face():
    frame = np.full((206, 184, 3), (128, 150, 182), np.uint8)
    mesh = np.full((478, 2), (92, 116), np.float32)
    theta = np.linspace(0, 2 * np.pi, len(LOWER_FACE), endpoint=False)
    mesh[LOWER_FACE] = np.column_stack(
        [92 + 68 * np.cos(theta), 127 + 52 * np.sin(theta)]
    )
    mesh[33], mesh[263] = (49, 76), (133, 76)
    mesh[61], mesh[291] = (71, 126), (113, 126)
    return frame, mesh


def displaced_face(frame):
    damaged = frame.copy()
    cv2.fillPoly(
        damaged, [np.array([[28, 88], [92, 100], [141, 161], [60, 164]])], (19, 26, 37)
    )
    for x in range(36, 101, 5):
        cv2.line(damaged, (x, 101), (x + 36, 150), (99, 98, 110), 1)
    return damaged


def test_displaced_cheek_fails_while_new_lip_shapes_pass():
    frame, mesh = reference_face()
    speech = frame.copy()
    cv2.ellipse(speech, (92, 126), (19, 7), 0, 0, 360, (19, 28, 31), -1)
    assert not integrity._suspect(integrity._cheek_metrics(frame, speech, mesh))

    damaged = displaced_face(speech)
    assert integrity._suspect(integrity._cheek_metrics(frame, damaged, mesh))
    assert not integrity._suspect(integrity._cheek_metrics(frame, frame, mesh))


def test_decodes_inserted_frames_at_delivery_rate(tmp_path):
    frame, face = reference_face()
    raw = tmp_path / "actor.avi"
    master = tmp_path / "master.avi"
    for file, fps in ((raw, 12), (master, 24)):
        output = cv2.VideoWriter(
            str(file), cv2.VideoWriter_fourcc(*"MJPG"), fps, (184, 206)
        )
        assert output.isOpened()
        for index in range(fps):
            output.write(
                displaced_face(frame) if file == master and index in (3, 5) else frame
            )
        output.release()
    report = integrity.assess(raw, master, mesh=np.repeat(face[None, ...], 12, axis=0))
    assert report["frames_checked"] == 24
    assert report["failure_times"] == [0.125, 0.2083]
    assert report["cheek_failure_times"] == [0.125, 0.2083]
    assert report["lip_texture"]["frames_checked"] == 24
    assert len(report["frame_measurements"][3]["lip_texture"]) > 1


def test_rejects_a_mouth_texture_defect_with_intact_cheeks(tmp_path, monkeypatch):
    frame, mesh = reference_face()
    source = tmp_path / "reference.avi"
    writer = cv2.VideoWriter(
        str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (184, 206)
    )
    assert writer.isOpened()
    for _ in range(5):
        writer.write(frame)
    writer.release()
    samples = iter((False, True, True, False, False))
    monkeypatch.setattr(
        integrity,
        "_texture_metrics",
        lambda *args: {"pale_contraction": next(samples), "foreign_dark_patch": False},
    )
    report = integrity.assess(source, source, mesh=np.repeat(mesh[None], 5, axis=0))
    assert report["version"] == 5 and not report["passed"]
    assert report["lip_texture"]["version"] == 3
    assert report["cheek_failure_times"] == []
    assert report["failure_times"] == [0.1, 0.2]
    assert report["lip_texture"]["defects"]["pale_contraction"] == [0.1, 0.2]
