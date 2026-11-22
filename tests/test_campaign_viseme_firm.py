import numpy as np
import pytest

from framecraft.campaign_viseme_firm import (
    _performance_index,
    _repair_envelope,
    _rounding_ratios,
    repair_neural,
)


def token(letter, time, confidence=0.99):
    return {"letter": letter, "time": time, "confidence": confidence}


def test_short_closed_consonant_repair_preserves_distant_neural_frames():
    original = np.tile([0.09, 0.5], (50, 1))
    posed = original.copy()
    posed[11:14, 0] = 0.008
    tokens = [token("m", 0.51), token("e", 1.30), token("b", 1.6, 0.20)]
    envelope, changes = _repair_envelope(original, posed, 25, tokens)
    assert changes == [{"letter": "m", "time": 0.51}]
    assert envelope.max() == 1
    assert np.all(envelope[:9] == 0)
    assert np.all(envelope[16:] == 0)
    identical, _ = _repair_envelope(original, original, 25, tokens)
    assert not identical.any()


def test_rounding_does_not_distort_already_rounded_vowels_or_other_frames():
    geo = np.tile([0.08, 0.55], (100, 1))
    tokens = [token("a", 0.6), token("ü", 1.2), token("i", 1.8)]
    ratios = _rounding_ratios(geo, tokens, 25)
    assert 0.82 < ratios[30] < 0.90
    assert np.all(ratios[:26] == 1)
    assert np.all(ratios[34:] == 1)
    geo[26:35, 1] = 0.45
    assert np.all(_rounding_ratios(geo, tokens, 25) == 1)
    assert np.all(_rounding_ratios(geo, [token("a", 0.5)], 25) == 1)


def test_signed_acoustic_visual_offset_uses_later_mouth_for_a_delayed_source():
    assert _performance_index(0.4, 25, 50, -0.12) == 13
    assert _performance_index(0.4, 25, 50, 0.12) == 7
    assert _performance_index(0.0, 25, 50, 0.12) == 0
    assert _performance_index(2.0, 25, 50, -0.12) == 49
    with pytest.raises(ValueError, match="timing"):
        repair_neural("a", "b", [], "c", offset=1.5)


def test_texture_repair_requires_same_source_audio_timing_and_actual_failed_video(
    tmp_path, monkeypatch
):
    import hashlib

    import framecraft.campaign_viseme_firm as firm

    video, donor = tmp_path / "source.mp4", tmp_path / "donor.mp4"
    video.write_bytes(b"presenter")
    donor.write_bytes(b"donor")
    sig = hashlib.sha256(video.read_bytes()).hexdigest()
    receipt = {
        "video_sha256": sig,
        "source_sha256": "camera-signature",
        "audio_sha256": "voice-signature",
        "source_start": 1,
        "voice_onset": 0.15,
        "seconds": 2.0,
    }
    dr = dict(receipt, video_sha256=hashlib.sha256(donor.read_bytes()).hexdigest())
    report = {
        "video_sha256": sig,
        "lip_texture": {"version": 2, "defects": {"pale_contraction": [0.67, 0.7]}},
    }
    seen = []

    def renderer(v, d, tokens, times_target, out):
        seen.extend(times_target)
        return {}

    monkeypatch.setattr(
        firm,
        "restore_lip_texture",
        lambda v, d, times, tokens, out: renderer(v, d, tokens, times, out),
    )
    firm.repair_from_report(
        video,
        donor,
        [],
        report,
        tmp_path / "result.mp4",
        video_receipt=receipt,
        donor_receipt=dr,
    )
    assert seen == [0.67, 0.7]
    with pytest.raises(ValueError, match="camera take"):
        firm.repair_from_report(
            video,
            donor,
            [],
            report,
            tmp_path / "result.mp4",
            video_receipt=receipt,
            donor_receipt=dict(dr, audio_sha256="other"),
        )
    with pytest.raises(ValueError, match="timing"):
        firm.repair_from_report(
            video,
            donor,
            [],
            report,
            tmp_path / "result.mp4",
            video_receipt=receipt,
            donor_receipt=dict(dr, voice_onset=0.4),
        )
    video.write_bytes(b"new")
    with pytest.raises(ValueError, match="different video bytes"):
        firm.repair_from_report(
            video,
            donor,
            [],
            report,
            tmp_path / "result.mp4",
            video_receipt=receipt,
            donor_receipt=dr,
        )


def test_eye_registration_restores_tissue_when_target_lip_corners_collapse():
    from framecraft.campaign_viseme_firm import _head_registration

    original = np.zeros((478, 2), np.float32)
    original[[33, 263]] = [[20, 30], [100, 30]]
    original[1] = [60, 54]
    original[[61, 291]] = [[34, 70], [86, 70]]
    narrow = original.copy()
    narrow[[61, 291]] = [[53, 72], [67, 72]]
    matrix = _head_registration(original, narrow)
    assert matrix is not None
    transformed = np.column_stack([original[[61, 291]], np.ones(2)]) @ matrix.T
    np.testing.assert_allclose(transformed.mean(0), [60, 72], atol=0.001)
    assert 40 < np.linalg.norm(transformed[1] - transformed[0]) < 52
    different_view = narrow.copy()
    different_view[1] = [0, 54]
    assert _head_registration(original, different_view) is None
