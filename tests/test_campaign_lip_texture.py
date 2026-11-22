from pathlib import Path

import numpy as np
import pytest

from framecraft.campaign_lip_texture import _patch_metrics, _texture_verdict


@pytest.mark.parametrize(
    ("name", "defect"),
    [
        ("pale", "pale_contraction"),
        ("hair", "foreign_dark_patch"),
        ("erased", "upper_lip_texture_erased"),
        ("unchanged_beard", None),
        ("existing_hairline", None),
        ("normal", None),
        ("teeth", None),
    ],
)
def test_decoded_source_matched_performances(name, defect):
    # Rectified perioral patches from actual same-actor original/rendered frames.
    # The unmodified source, natural bilabial closure and a tooth highlight are
    # controls for the pale contraction and transferred hair defects.
    path = Path(__file__).parent / "fixtures" / "presenter-lip-texture.npz"
    with np.load(path) as clips:
        original, edited = clips[name + "_source"], clips[name + "_edit"]
    rows = [
        {"time": i / 30, "lip_texture": _patch_metrics(a, b)}
        for i, (a, b) in enumerate(zip(original, edited))
    ]
    report = _texture_verdict(rows)
    assert report["passed"] == (defect is None)
    if defect:
        assert report["defects"][defect]
    assert _texture_verdict(
        [
            {"time": i / 30, "lip_texture": _patch_metrics(a, a)}
            for i, a in enumerate(original)
        ]
    )["passed"]
