import cv2
import numpy as np
import pytest

from framecraft.campaign_lip_composite import INNER, OUTER, alignment, composite


def mesh(inner_height):
    p = np.zeros((478, 2), np.float32)
    for idx, value in {
        33: (16, 16),
        133: (28, 16),
        362: (52, 16),
        263: (64, 16),
        168: (40, 18),
        6: (40, 22),
        197: (40, 25),
        1: (40, 30),
        98: (34, 30),
        327: (46, 30),
    }.items():
        p[idx] = value
    # Start at the left corner, with upper-lip vertices before the right corner.
    a = np.linspace(-np.pi, np.pi, len(OUTER), endpoint=False)
    p[OUTER] = np.stack(
        [40 + 13 * np.cos(a), 43 + (inner_height + 3) * np.sin(a)], axis=1
    )
    p[INNER] = np.stack([40 + 10 * np.cos(a), 43 + inner_height * np.sin(a)], axis=1)
    return p


def test_only_inner_cavity_accepts_generated_pixels():
    x, y = np.meshgrid(np.arange(84), np.arange(70))
    stipple = (60 + 80 * ((x + 2 * y) % 3 == 0)).astype(np.uint8)
    source = np.stack([stipple] * 3, axis=2)
    original, spoken = mesh(1.3), mesh(4.4)
    cv2.fillPoly(source, [np.rint(original[OUTER]).astype(np.int32)], (40, 60, 185))
    cv2.fillPoly(source, [np.rint(original[INNER]).astype(np.int32)], (18, 18, 18))
    generated = np.full_like(source, (255, 245, 0))
    cv2.fillPoly(generated, [np.rint(spoken[INNER]).astype(np.int32)], (240, 2, 3))
    result = composite(source, generated, original, spoken)
    assert tuple(result[43, 40]) == (240, 2, 3)
    np.testing.assert_array_equal(result[:32], source[:32])
    np.testing.assert_array_equal(result[55:], source[55:])
    np.testing.assert_array_equal(result[:, :22], source[:, :22])
    np.testing.assert_array_equal(result[:, 59:], source[:, 59:])
    assert result[37, 40, 2] > result[37, 40, 0] + 60
    shaded = composite(source, generated, original, spoken, lip_shading=0.78)
    np.testing.assert_array_equal(shaded[:32], source[:32])
    np.testing.assert_array_equal(shaded[55:], source[55:])
    np.testing.assert_array_equal(shaded[:, :22], source[:, :22])
    np.testing.assert_array_equal(shaded[43, 40], result[43, 40])
    assert np.any(shaded[37, 40] != result[37, 40])


def test_alignment_rejects_different_upper_face():
    source = mesh(1)
    other = source.copy()
    other[33, 1] += 29
    with pytest.raises(ValueError, match="match"):
        alignment(source, other)
