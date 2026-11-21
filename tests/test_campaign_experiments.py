import hashlib
import wave

import numpy as np
import pytest

from framecraft.campaign_experiments import assemble_voice, export, matrix
from framecraft.core import save


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
