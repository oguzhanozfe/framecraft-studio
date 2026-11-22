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
