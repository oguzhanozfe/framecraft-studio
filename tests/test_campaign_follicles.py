import cv2
import numpy as np

from framecraft.campaign_follicles import UPPER, preserve_detail


def test_original_hairs_return_without_replacing_speaking_lips():
    actual = np.full((100, 130, 3), 152, dtype=np.uint8)
    for x in range(44, 87, 3):
        cv2.line(actual, (x, 48), (x - 1, 56), (38, 53, 62), 1)
    base = cv2.GaussianBlur(actual, (13, 13), 3)
    base[62:69, 43:89] = [80, 85, 173]
    landmarks = np.zeros((478, 2), np.float32)
    landmarks[[33, 263]] = [[23, 29], [107, 29]]
    landmarks[2] = [65, 45]
    landmarks[1] = [65, 42]
    landmarks[UPPER] = [
        [42, 65],
        [48, 62],
        [55, 61],
        [65, 61],
        [74, 61],
        [81, 62],
        [88, 65],
    ]
    restored = preserve_detail(base, landmarks, actual, landmarks)
    assert np.array_equal(restored[62:69, 53:78], base[62:69, 53:78])
    assert np.array_equal(restored[:39], base[:39])
    band = np.s_[46:58, 44:87]
    error = lambda a: np.mean(cv2.absdiff(a[band], actual[band]))
    assert error(restored) < error(base) * 0.70
    assert preserve_detail(actual, landmarks, actual, landmarks) is actual


def test_measured_second_pass_leaves_frames_outside_failure_window(
    monkeypatch, tmp_path
):
    from framecraft import campaign_follicles as engine

    scene = [np.full((40, 40, 3), n, dtype=np.uint8) for n in range(24)]
    mesh = np.zeros((24, 478, 2), np.float32)

    class Capture:
        def get(self, prop):
            return 24 if prop == cv2.CAP_PROP_FPS else len(scene)

        def set(self, *args):
            return True

        def read(self):
            return True, scene[0]

        def release(self):
            pass

    drawn = []
    monkeypatch.setattr(engine, "_take", lambda _: (scene, mesh, 24))
    monkeypatch.setattr(engine.cv2, "VideoCapture", lambda _: Capture())
    monkeypatch.setattr(engine, "track", lambda *args: mesh)
    monkeypatch.setattr(
        engine, "preserve_detail", lambda base, *args, measured_weight=None: base.copy()
    )
    monkeypatch.setattr(
        engine, "_encode", lambda _, fps, size, frames: drawn.extend(list(frames))
    )
    result = engine.restore(
        tmp_path / "original",
        tmp_path / "take",
        tmp_path / "out",
        source_start=0,
        failure_times=[0.5],
    )
    assert 1 < result["restored_frames"] < 7
    assert drawn[0] is scene[0]
    assert drawn[12] is not scene[12]
    assert drawn[-1] is scene[-1]
