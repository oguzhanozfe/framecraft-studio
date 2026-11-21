import cv2
import numpy as np
import pytest

from framecraft import campaign
from framecraft.campaign_edit import FPS, moving_frames, preview_cfr
from framecraft.core import read, save
from framecraft.edit import ffmpeg


def test_every_edit_is_exactly_thirty_seconds():
    for index in range(12):
        clock = campaign.timeline(index)
        assert sum(duration * FPS for duration, _ in clock) == 720
        assert [duration for duration, host in clock if host] == [4, 4]
        assert len(clock) == 6
    for layout, hosts in [("gameplay", 0), ("creator_open", 1), ("creator_close", 1)]:
        clock = campaign.timeline(0, {"layout": layout})
        assert sum(s for s, _ in clock) == 30
        assert sum(h for _, h in clock) == hosts


def test_short_clip_requires_explicit_loop(tmp_path):
    clip = tmp_path / "short.avi"
    recorder = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"MJPG"), 24, (96, 96))
    for level in range(6):
        recorder.write(np.full((96, 96, 3), level * 30, np.uint8))
    recorder.release()
    with pytest.raises(ValueError, match="needs"):
        list(moving_frames(clip, 8))
    seq = [int(f[640, 360, 0]) for f in moving_frames(clip, 8, allow_loop=True)]
    assert seq[:2] == seq[6:]


def test_fact_editor_cannot_approve_a_verbatim_rejected_claim(tmp_path, monkeypatch):
    beats = [
        {"voice": f"Original claim number {n}", "caption": f"Cue {n}"} for n in range(6)
    ]
    save(tmp_path / "copy.json", {"beats": beats})
    item = dict(
        name="product", proof="one fan", limits="no benchmark", audience="PC builders"
    )

    def model(schema, *args):
        if schema is campaign.FactCheck:
            return schema(
                revisions=[
                    {
                        "beat": 1,
                        "reason": "unsupported claim",
                        "voice": beats[0]["voice"],
                    }
                ]
            )
        return schema(
            lines=[f"Product için yeni bir kaynak temelli cümle {n}" for n in range(6)]
        )

    monkeypatch.setattr(campaign, "ask", model)
    campaign.approve_copy(item, tmp_path, 0)
    assert (
        read(tmp_path / "approved-copy.json")["beats"][0]["voice"]
        == "Product için yeni bir kaynak temelli cümle 0"
    )


def test_spoken_product_sizes_must_match_source():
    assert campaign.quantities("200 ml ve 12 GB") == campaign.quantities(
        "iki yüz mililitrelik şişe ve on iki gigabayt"
    )
    assert campaign.quantities("iki litrelik") != campaign.quantities("200 ml")


def test_publisher_start_timestamp_is_not_part_of_the_content(tmp_path):
    ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=96x64:rate=24:duration=2",
            "-output_ts_offset",
            "10",
            "-c:v",
            "libx264",
            tmp_path / "preview.mp4",
        ]
    )
    cap = cv2.VideoCapture(str(preview_cfr(tmp_path)))
    try:
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 48
    finally:
        cap.release()


def test_claim_judge_preserves_the_verified_spoken_product_name(monkeypatch):
    item = {
        "name": "ASUS Dual GeForce RTX 5070 OC",
        "spoken_name": "ASUS Dual RTX 5070",
        "audience": "oyuncu",
        "proof": "İki fan, 12 GB bellek.",
        "limits": "FPS testi yok.",
    }
    lines = ["ASUS Dual RTX 5070 burada.", "Kasa hep soğuk kalır."]
    monkeypatch.setattr(
        campaign,
        "ask",
        lambda *a: campaign.ClaimReview(
            issues=[
                campaign.CopyFinding(
                    quotation=item["spoken_name"], reason="Kısa model adı."
                ),
                campaign.CopyFinding(
                    quotation="soğuk kalır", reason="Ölçüm yapılmamış."
                ),
            ]
        ),
    )
    assert campaign.verify_claims(item, lines) == ["soğuk kalır — Ölçüm yapılmamış."]
