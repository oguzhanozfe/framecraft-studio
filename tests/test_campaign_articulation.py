import numpy as np

from framecraft.campaign_articulation import assess_track
from framecraft.campaign_speech_anchors import _forced_tokens
from framecraft.campaign_speech_controls import compare_features, match_onset


def signals():
    fps = 30
    t = np.arange(120) / fps
    energy = sum(
        np.exp(-(((t - center) / radius) ** 2)) * level
        for center, radius, level in [
            (0.45, 0.10, 0.8),
            (0.78, 0.06, 0.4),
            (1.14, 0.13, 0.7),
            (1.72, 0.095, 1.0),
            (1.99, 0.07, 0.3),
            (2.36, 0.10, 0.9),
            (2.85, 0.085, 0.55),
        ]
    )
    gap = 0.004 + 0.10 * energy**0.5
    track = np.column_stack([gap, 0.48 + 0.035 * energy])
    return track, energy, fps


def test_aligned_and_displaced_speech_with_identical_energy_and_face():
    tr, e, f = signals()
    assert assess_track(tr, e, f)["passed"]
    moved = np.roll(tr, 10, axis=0)
    report = assess_track(moved, e, f)
    assert not report["passed"]
    assert "displaced_speech_rhythm" in report["failures"]


def test_static_and_untracked_mouth_cannot_pass():
    tr, e, f = signals()
    motionless = np.tile(tr.mean(axis=0), (len(e), 1))
    assert "weak_mouth_articulation" in assess_track(motionless, e, f)["failures"]
    tr[20:68] = np.nan
    assert assess_track(tr, e, f)["status"] == "inconclusive"


def test_speaking_while_sound_is_silent():
    tr, e, f = signals()
    silence = e < 1e-6
    tr[silence, 0] = np.where(np.arange(silence.sum()) % 2, 0.11, 0.01)
    assert "mouth_moves_through_silence" in assess_track(tr, e, f)["failures"]


def test_vowel_shaped_mouth_during_bilabials_fails_even_when_rhythm_matches():
    tr, e, f = signals()
    trusted = [
        {"time": 0.63, "type": "bilabial", "letter": "m"},
        {"time": 0.45, "type": "open_vowel", "letter": "a"},
        {"time": 1.14, "type": "open_vowel", "letter": "a"},
    ]
    assert assess_track(tr, e, f, trusted)["passed"]
    wrong = [{**x, "time": 1.72} if x["type"] == "bilabial" else x for x in trusted]
    report = assess_track(tr, e, f, wrong)
    assert (
        "bilabial_mismatch" in report["failures"]
        and report["metrics"]["bilabial_fraction"] == 0
    )


def test_forced_alignment_keeps_separate_spikes_for_a_repeated_letter():
    path = [0, 1, 2, 0, 2, 3, 0]
    probs = np.full((len(path), 4), -10.0)
    probs[np.arange(len(path)), path] = 0
    result = _forced_tokens(probs, [1, 2, 2, 3], 0)
    assert [(i, indices.tolist()) for i, indices in result] == [
        (1, [1]),
        (2, [2]),
        (2, [4]),
        (3, [5]),
    ]


def test_true_phonetic_shapes_disambiguate_a_repeated_loudness_peak():
    fps = 30
    t = np.arange(145) / fps
    lip = np.full(t.shape, 0.003)
    openings = [0.88, 1.23, 1.65, 2.37, 2.65, 3.15]
    closures = [1.06, 1.42, 1.99, 2.5, 2.91, 3.32]
    anchors = [{"time": x, "type": "open_vowel", "letter": "a"} for x in openings]
    anchors.extend({"time": x, "type": "bilabial", "letter": "m"} for x in closures)
    for instant in openings:
        lip += 0.102 * np.exp(-(((t - instant) / 0.088) ** 2))
    track = np.column_stack([lip, 0.46 + 0.02 * lip])
    # This stimulus has a strong acoustic energy peak 300 ms before each pictured
    # opening. Its independently known speech events specify the true alignment.
    alias = 0.00005 + np.roll((lip / 0.13) ** 2, -9)
    assert not assess_track(track, alias, fps)["passed"]
    assert assess_track(track, alias, fps, anchors)["passed"]
    wrong = np.roll(track, -9, axis=0)
    assert assess_track(wrong, alias, fps)["passed"]
    assert not assess_track(wrong, alias, fps, anchors)["passed"]


def test_same_face_and_actual_line_must_outperform_different_line_and_derangements():
    rng = np.random.default_rng(81)
    true = rng.normal(size=(44, 16))
    wrong = rng.normal(size=true.shape)
    visible = true + rng.normal(scale=0.14, size=true.shape)
    speech = np.ones(44, dtype=bool)
    assert compare_features(visible, true, wrong, speech)["status"] == "pass"
    assert compare_features(visible, wrong, true, speech)["status"] == "fail"
    assert compare_features(visible, true, true, speech)["status"] == "inconclusive"
    static = np.zeros_like(visible)
    assert (
        compare_features(static, true, wrong + 8, speech)["reason"]
        == "indistinguishable_syllable_order"
    )
    assert (
        compare_features(visible, true, wrong, np.arange(44) < 8)["status"]
        == "inconclusive"
    )


def test_counterfactual_has_same_onset_and_level_as_true_line():
    true = np.zeros(16000, dtype=np.int16)
    true[3200:11200] = 6000
    foil = np.zeros(16000, dtype=np.int16)
    foil[800:8800] = -2000
    bed = match_onset(true, foil)
    np.testing.assert_array_equal(bed, -true)
    assert match_onset(true, np.zeros(16000, dtype=np.int16)) is None
