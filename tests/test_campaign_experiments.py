import hashlib
import wave

import numpy as np
import pytest

from framecraft.campaign_experiments import assemble_voice, export, matrix
from framecraft.core import save


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
        save(folder / "final-lips.json", {"passed": True, "sha256": sig})
        rows.append(row)
    dest = tmp_path / "package"
    assert export(tmp_path, catalog, dest) == 2
    assert len(list(dest.glob("*.mp4"))) == 2
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
