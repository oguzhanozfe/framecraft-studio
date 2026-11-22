import hashlib
import wave

import numpy as np
import pytest

from framecraft.campaign_experiments import assemble_voice, export, matrix
from framecraft.campaign_face_integrity import FACE_INTEGRITY_VERSION
from framecraft.core import save


def test_superseded_source_cannot_replace_an_accepted_edit(tmp_path, monkeypatch):
    from framecraft import campaign_experiments as experiment

    root = tmp_path / "item"
    work = root / "H1-C1"
    work.mkdir(parents=True)
    save(root / "required-picture.json", {"report": "approved.json"})
    save(root / "approved.json", {"passed": True, "sha256": "current"})
    save(work / "delivery.json", {"sha256": "reviewed-file"})
    monkeypatch.setattr(
        experiment, "verified_master", lambda path: ({}, {"sha256": "old"})
    )
    with pytest.raises(ValueError, match="superseded"):
        experiment.edition(tmp_path, {"slug": "item", "base": "old/motion.json"}, 0, 0)
    from framecraft.core import read

    assert read(work / "delivery.json") == {"sha256": "reviewed-file"}


def test_reattached_product_take_keeps_its_review_and_adjudicates_once(
    tmp_path, monkeypatch
):
    from framecraft import motion_review

    monkeypatch.setenv("FRAMECRAFT_HOME", str(tmp_path / "state"))
    calls = []

    def inspect(video, reference, signature, model, disputes):
        calls.append(video.name)
        row = {"input": list(signature), "defect": "warped_geometry"}
        if disputes:
            row.update(defect="none", adjudication=list(disputes))
        return row

    monkeypatch.setattr(motion_review, "_inspect", inspect)
    a, b, ref = [tmp_path / s for s in ("first.mp4", "reattached.mp4", "item.jpg")]
    for f in (a, b, ref):
        f.write_bytes(b"exact content")
    original = motion_review.reference_check(a, ref, model="local")
    assert motion_review.reference_check(b, ref, model="local") == original
    conflicting = [original, {**original, "defect": "none"}]
    for f in (a, b):
        assert (
            motion_review.reference_check(
                f, ref, model="local", disputed_checks=conflicting
            )["defect"]
            == "none"
        )
    assert len(calls) == 2


def test_decode_comparison_catches_an_uncontrolled_picture_change(
    tmp_path, monkeypatch
):
    from framecraft import campaign_pairqa as qa

    files = [tmp_path / n for n in ("a.mp4", "b.mp4")]
    for file in files:
        file.write_bytes(file.name.encode())
    first = np.zeros((3, 8), np.uint8)
    second = first.copy()
    second[-1] = 10
    monkeypatch.setattr(qa, "picture", lambda f: first if f == files[0] else second)
    monkeypatch.setattr(qa, "sound", lambda f: np.ones(8000, np.float32))
    assert qa.compare(*files, 2, tmp_path / "one.json")["passed"]
    second[0] = 10
    assert not qa.compare(*files, 2, tmp_path / "two.json")["passed"]


def test_print_receipt_is_bound_to_the_exact_file_source_and_interval(tmp_path):
    from framecraft.campaign_experiments import verified_print
    from framecraft.reference_detail import print_crop

    work = tmp_path / "H1-C1"
    work.mkdir()
    cue = dict(
        report="heel.json",
        phrase="Brand",
        reference_sha256="photo",
        start_frame=480,
        frames=150,
        minimum_fraction=0.9,
    )
    receipt = dict(
        passed=True,
        video_sha256="master",
        source_sha256="photo",
        start_frame=480,
        frames=150,
        matches={"Brand": 150},
    )
    save(work.parent / "required-print.json", [cue])
    assert not verified_print(work, "master")
    save(work / "heel.json", receipt)
    assert verified_print(work, "master")
    assert not verified_print(work, "old-master")
    for wrong in (
        {"start_frame": 479},
        {"source_sha256": "other"},
        {"matches": {"Brand": 134}},
    ):
        save(work / "heel.json", {**receipt, **wrong})
        assert not verified_print(work, "master")
    left, top, right, bottom = print_crop((1000, 1500), (0.8, 0.5, 0.9, 0.55))
    assert 0 <= left <= 1200 < 1350 <= right <= 1500
    assert 0 <= top <= 500 < 550 <= bottom <= 1000


def test_repeat_review_locates_both_copies_of_a_moving_sequence():
    from framecraft.campaign_motion import repeated_windows

    samples = np.arange(70, dtype=np.uint8)[:, None] * 3
    samples[38:48] = samples[11:21]
    assert repeated_windows(samples) == (11, 38)
    assert repeated_windows(np.full((40, 3), 90)) is None


def test_twelve_brands_produce_25_matched_pairs():
    products = [{"slug": str(i), "hooks": 3 if i == 0 else 2} for i in range(12)]
    variants = list(matrix(products))
    assert len(variants) == 50
    for p in products:
        for h in range(p["hooks"]):
            assert [c for sku, n, c in variants if sku == p and n == h] == [0, 1]


def test_voice_placement_respects_measured_offset(tmp_path):
    data = np.full(2400, 1100, dtype="<i2")
    with wave.open(str(tmp_path / "hook.wav"), "wb") as file:
        file.setparams((1, 2, 24000, 0, "NONE", ""))
        file.writeframes(data.tobytes())
    target = tmp_path / "mix.wav"
    assemble_voice(tmp_path, [(0.37, "hook", "Ses")], target, 1)
    with wave.open(str(target), "rb") as file:
        result = np.frombuffer(file.readframes(file.getnframes()), dtype="<i2")
    assert len(result) == 24000 and np.flatnonzero(result)[0] == 8880
    with pytest.raises(ValueError, match="does not fit"):
        assemble_voice(tmp_path, [(0.1, "hook", "a"), (0.14, "hook", "b")], target, 1)


def test_export_requires_two_verified_distinct_variants(tmp_path):
    catalog = {"products": [{"slug": "item", "hooks": 1}], "require_final_lips": True}
    rows = []
    for c, body in enumerate([b"one", b"two"], 1):
        folder = tmp_path / "item" / f"H1-C{c}"
        folder.mkdir(parents=True)
        src = folder / "master.mp4"
        src.write_bytes(body)
        sig = hashlib.sha256(body).hexdigest()
        row = {
            "slug": "item",
            "project": "Project",
            "audience": "players",
            "hook": 1,
            "cta": c,
            "hook_text": "hook",
            "cta_text": str(c),
            "archive": str(src),
            "sha256": sig,
        }
        save(folder / "delivery.json", row)
        checks = {
            "opening": {"passed": True, "version": FACE_INTEGRITY_VERSION},
            "closing": {"passed": True, "version": FACE_INTEGRITY_VERSION},
        }
        save(
            folder / "final-lips.json",
            {
                "version": 2,
                "passed": True,
                "sha256": sig,
                "articulation": checks,
                "spoken_identity": checks,
                "face_integrity": checks,
            },
        )
        rows.append(row)
    dest = tmp_path / "package"
    assert export(tmp_path, catalog, dest) == 2
    assert len(list(dest.glob("*.mp4"))) == 2
    save(
        tmp_path / "item/H1-C2/final-lips.json",
        {"passed": True, "sha256": rows[1]["sha256"]},
    )
    with pytest.raises(ValueError, match="audiovisual"):
        export(tmp_path, catalog, dest)

    save(
        tmp_path / "item" / "H1-C2" / "final-lips.json",
        {"passed": False, "sha256": rows[1]["sha256"]},
    )
    with pytest.raises(ValueError, match="audiovisual"):
        export(tmp_path, catalog, dest)
    assert export(tmp_path, catalog, tmp_path / "partial", ready_only=True) == 0
    assert not list((tmp_path / "partial").glob("*.mp4"))
    import os

    was_linked = os.path.samefile(dest / "item-H1-A.mp4", rows[0]["archive"])
    assert export(tmp_path, catalog, dest, ready_only=True) == 0
    if was_linked:
        assert not list(dest.glob("*.mp4"))
    assert (tmp_path / "item/H1-C1/master.mp4").is_file()


def test_cast_binds_measured_onset_to_picture_and_voice(tmp_path, monkeypatch):
    from framecraft import campaign_sync
    from framecraft.campaign_cast import prepare_take

    audio = tmp_path / "hook0.wav"
    with wave.open(str(audio), "wb") as w:
        w.setparams((1, 2, 24000, 0, "NONE", ""))
        w.writeframes(np.full(2400, 900, dtype="<i2").tobytes())
    source = tmp_path / "selfie.mp4"
    source.write_bytes(b"a")
    output = tmp_path / "hook0-face.mp4"
    onsets, renders = [], []

    class Network:
        def run(self, source, bed, video, *args, **kwargs):
            renders.append(source)
            video.write_bytes(source.read_bytes() + b"face")

    def measure(video, bed, *args, **kwargs):
        with wave.open(str(bed), "rb") as w:
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        onset = np.flatnonzero(pcm)[0] / 24000
        onsets.append(onset)
        lag = 0 if onset == 0.37 else -120
        return {"passed": lag == 0, "offset_ms": lag, "confidence": 1.1}

    monkeypatch.setattr(campaign_sync, "measure", measure)
    from framecraft import campaign_face_integrity, campaign_speech_review

    monkeypatch.setattr(
        campaign_speech_review, "review", lambda *a, **kw: {"passed": True}
    )
    monkeypatch.setattr(
        campaign_face_integrity,
        "assess",
        lambda *a, **kw: {"passed": True, "version": FACE_INTEGRITY_VERSION},
    )
    args = (Network(), source, audio, output, tmp_path, tmp_path)
    assert prepare_take(*args, start=0, seconds=1)["voice_onset"] == 0.37
    assert prepare_take(*args, start=0, seconds=1)["offset_ms"] == 0
    assert len(renders) == 1 and onsets == [0.25, 0.37]
    source.write_bytes(b"b")
    assert prepare_take(*args, start=0, seconds=1)["passed"]
    assert len(renders) == 2 and output.read_bytes() == b"bface"


def test_product_print_checks_every_decoded_frame(tmp_path, monkeypatch):
    import gzip
    import json

    import cv2

    from framecraft import campaign_print

    name = "SEA SALT"
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    reference = tmp_path / "reference.png"
    cv2.imwrite(str(reference), image)
    video = tmp_path / "shot.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 64))
    for _ in range(4):
        writer.write(image)
    writer.release()
    results = iter((name, name, "SER SALT", name, name))
    monkeypatch.setattr(
        campaign_print,
        "read_text",
        lambda _: [{"text": next(results), "confidence": 0.99}],
    )
    report = tmp_path / "print.json"
    qa = campaign_print.verify(reference, video, [name], report)
    assert not qa["passed"] and qa["matches"][name] == 3 and qa["frames"] == 4
    with gzip.open(tmp_path / qa["ledger"], "rt") as f:
        rows = [json.loads(row) for row in f]
    assert [r["frame"] for r in rows if not r["readable"][name]] == [1]


def test_reference_repair_keeps_its_crop_inside_the_catalog_photo(
    tmp_path, monkeypatch
):
    import cv2

    from framecraft import motion_review
    from framecraft.core import read

    photo = tmp_path / "source.png"
    cv2.imwrite(str(photo), np.zeros((100, 60, 3), dtype=np.uint8))
    direction = "Follow the original seam while the fabric moves gently."
    view = motion_review.DetailView(
        center_x=0.98, center_y=0.1, fraction=0.6, action=direction
    )
    monkeypatch.setattr(motion_review, "ask", lambda *args, **kwargs: view)
    result = tmp_path / "detail.png"
    assert (
        motion_review.repair_view(photo, result, "Show the seam", "Strap drifted")
        == direction
    )
    assert cv2.imread(str(result)).shape[:2] == (100, 60)
    assert read(result.with_suffix(".json"))["box"] == [24, 0, 60, 60]


def test_candidate_with_the_wrong_timed_wav_is_rejected(tmp_path):
    from framecraft.campaign_cast import adopt_candidate, digest, voice_bed

    actor, candidate = tmp_path / "actor.mp4", tmp_path / "candidate.mp4"
    actor.write_bytes(b"original")
    candidate.write_bytes(b"candidate")
    voice = tmp_path / "hook0.wav"
    with wave.open(str(voice), "wb") as out:
        out.setparams((1, 2, 24000, 0, "NONE", ""))
        out.writeframes(np.full(2400, 1500, dtype="<i2").tobytes())
    bed = tmp_path / "timed.wav"
    voice_bed(voice, bed, 1, 0.4)
    save(
        candidate.with_suffix(".json"),
        {
            "source_sha256": digest(actor),
            "audio_sha256": digest(voice),
            "video_sha256": digest(candidate),
            "voice_onset": 0.25,
            "seconds": 1,
        },
    )
    acquittal = tmp_path / "physical.json"
    save(
        acquittal,
        {"passed": True, "sha256": digest(candidate), "bed_sha256": digest(bed)},
    )
    with pytest.raises(ValueError, match="recorded line or onset"):
        adopt_candidate(actor, tmp_path, "hook0", candidate, acquittal, bed, None, None)


def test_export_uses_the_current_speaking_take(tmp_path):
    from framecraft.campaign_experiments import current_takes

    speech = tmp_path / "speech"
    work = tmp_path / "H1-C1"
    speech.mkdir()
    work.mkdir()
    good = {"passed": True, "video_sha256": "same"}
    save(
        work / "motion-assessment.json",
        {"lip_sync": {"opening": good, "closing": good}},
    )
    save(speech / "hook0-sync.json", good)
    save(speech / "cta0-sync.json", good)
    assert current_takes(work, 0, 0)
    save(speech / "cta0-sync.json", {**good, "video_sha256": "replacement"})
    assert not current_takes(work, 0, 0)


def test_running_render_keeps_signed_bytes_after_new_cast(tmp_path):
    from framecraft.campaign_experiments import fingerprint, talking_take

    face = tmp_path / "cta0-face.mp4"
    wav = tmp_path / "cta0.wav"
    wav.write_bytes(b"voice")

    def select(data):
        stage = tmp_path / "new.mp4"
        stage.write_bytes(data)
        stage.replace(face)
        save(
            tmp_path / "cta0-sync.json",
            {
                "passed": True,
                "video_sha256": fingerprint(face),
                "audio_sha256": fingerprint(wav),
            },
        )
        return talking_take(tmp_path, "cta0")[0]

    first, second = select(b"first take"), select(b"second take")
    assert first != second
    assert first.read_bytes() == b"first take"
    assert second.read_bytes() == b"second take"


@pytest.mark.parametrize("passes_timing", [True, False])
def test_original_lip_texture_is_rechecked_against_real_speech(
    tmp_path, monkeypatch, passes_timing
):
    from framecraft import (
        campaign_face_integrity,
        campaign_lip_composite,
        campaign_speech_review,
        campaign_sync,
    )
    from framecraft.campaign_cast import checked_picture

    original = tmp_path / "camera.mp4"
    generated = tmp_path / "performance.mp4"
    voice = tmp_path / "dialogue.wav"
    compared = []

    def assess(source, video, **kwargs):
        assert source == original
        return {
            "passed": video != generated,
            "failure_times": [0.63] if video == generated else [],
        }

    def physical(video, bed, **kwargs):
        assert bed == voice
        compared.append(video.name)
        return {"passed": True}

    def restore(source, candidate, output, **kwargs):
        assert (source, candidate) == (original, generated)
        output.write_bytes(b"restored")

    monkeypatch.setattr(campaign_face_integrity, "assess", assess)
    monkeypatch.setattr(campaign_speech_review, "review", physical)
    monkeypatch.setattr(campaign_lip_composite, "render", restore)
    monkeypatch.setattr(
        campaign_sync, "measure", lambda *a, **kw: {"passed": passes_timing}
    )
    args = (original, generated, voice, None, None, 0.0, 2.0, "bak", {"passed": True})
    if passes_timing:
        result, shape, timing, skin = checked_picture(*args)
        assert result.read_bytes() == b"restored" and all(
            x["passed"] for x in (shape, timing, skin)
        )
    else:
        with pytest.raises(
            ValueError, match="changed the recorded articulation or timing"
        ):
            checked_picture(*args)
    assert compared == ["performance.mp4", "original-lip-texture.mp4"]
