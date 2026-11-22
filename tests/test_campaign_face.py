import numpy as np

from framecraft.campaign_face import lip_mask, mouth_transform


def test_registration_maps_both_lip_corners_without_shear():
    donor = np.zeros((478, 2), dtype=np.float32)
    target = donor.copy()
    donor[[61, 291]] = [[48, 88], [72, 90]]
    target[[61, 291]] = [[43, 90], [71, 97]]
    matrix = mouth_transform(donor, target)
    for i in (61, 291):
        np.testing.assert_allclose(matrix @ [*donor[i], 1], target[i], atol=1e-4)
    linear = matrix[:, :2]
    assert np.linalg.det(linear) > 0
    np.testing.assert_allclose(
        linear.T @ linear, np.eye(2) * np.linalg.det(linear), atol=1e-5
    )


def test_incompatible_mouth_pose_and_scale_are_rejected():
    face = np.zeros((478, 2), dtype=np.float32)
    face[291] = [30, 0]
    for corner in ([0, 25], [-20, 0], [60, 0], [0, 0]):
        other = face.copy()
        other[291] = corner
        assert mouth_transform(face, other) is None


def test_rest_mask_protects_eyes_and_cheeks():
    face = np.zeros((478, 2), np.float32)
    face[[33, 263]] = [[35, 35], [85, 35]]
    indices = [61, 40, 37, 0, 267, 270, 291, 321, 314, 17, 84, 91]
    face[indices] = [
        [45, 72],
        [49, 70],
        [55, 69],
        [60, 69],
        [65, 69],
        [71, 70],
        [75, 72],
        [71, 74],
        [65, 75],
        [60, 75],
        [55, 75],
        [49, 74],
    ]
    mask = lip_mask(face, (0, 0, 120, 120))[:, :, 0]
    assert mask[72, 60] > 0.95
    assert mask[:50].max() == 0
    assert mask[:, :31].max() == mask[:, 89:].max() == 0
