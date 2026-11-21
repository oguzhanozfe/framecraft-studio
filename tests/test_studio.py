import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from framecraft import catalog, pipeline, publish
from framecraft.audit import audit, judge, text_checks
from framecraft.core import Brief, create_job, jobdir, query, read, save
from framecraft.director import angles
from framecraft.edit import ffmpeg, plate
from framecraft.preview import PreviewMonitor
from framecraft.providers import missing, template
from framecraft.quality import glyph_quality
from framecraft.server import app


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("FRAMECRAFT_HOME", str(tmp_path / "studio"))
    monkeypatch.setenv("FRAMECRAFT_DIRECTOR", "off")


def test_cpu_frame_ocr_and_pts(tmp_path):
    clip = tmp_path / "clip.mp4"
    ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=256x384:r=12",
            "-t",
            "0.5",
            "-c:v",
            "libx264",
            clip,
        ]
    )
    report = audit(
        clip,
        tmp_path / "qa",
        "master",
        [{"start": 0, "end": 1, "text": "Missing text"}],
    )
    rows = [
        json.loads(r)
        for r in (tmp_path / "qa" / "frames.jsonl").read_text().splitlines()
    ]
    assert len(rows) == report["frames"] == 6
    assert rows[-1]["time"] == pytest.approx(5 / 12, abs=0.0001)
    assert set(report["blockers"]) >= {"caption_mismatch", "black"}


def test_printed_accents_and_single_frame_error(tmp_path):
    path = tmp_path / "plate.png"
    plate("Işık ve özgürlük", (576, 1024), path)
    image = cv2.imread(str(path))
    assert glyph_quality(image, path) > 0.96
    image[600:860] = 0
    assert glyph_quality(image, path) == 0
    assert "caption_glyphs" in judge({"caption_glyphs": 1}, 144)[1]


def test_text_zone_and_spelling():
    lines = [
        {"text": "Sen de kesfet", "confidence": 0.99, "box": [0.2, 0.65, 0.72, 0.71]}
    ]
    assert text_checks(lines, "Sen de keşfet", "master") == []
    assert text_checks(lines, "Başka metin", "master") == ["caption_mismatch"]
    assert text_checks(lines, "", "raw") == ["unplanned_text"]


def test_two_bad_attempts_never_promote(tmp_path, monkeypatch):
    j = create_job("create")
    save(
        jobdir(j) / "brief.json",
        Brief(brief="Test toy", shots=1, variants=1).model_dump(),
    )
    monkeypatch.setattr(
        pipeline, "inventory", lambda: {"models": [{"id": "h3", "available": True}]}
    )
    calls = []

    def fail(*args):
        calls.append(args[2])
        raise RuntimeError("renderer unavailable")

    monkeypatch.setattr(pipeline, "generate", fail)
    pipeline.make(j)
    assert len(set(calls)) == 2 and not (jobdir(j) / "winner.mp4").exists()
    assert read(jobdir(j) / "result.json")["winner"] is None
    assert (
        query("SELECT status FROM jobs WHERE id=?", (j,))[0]["status"] == "needs_review"
    )


def test_model_dependencies_and_chunks():
    assert "LTXV25ModelsLoader" in missing(template("ltx"), {})
    assert catalog.compatible("MiniMax H3") and not catalog.compatible("SDXL")
    for size in (13, 5 * 1024**2, 21 * 1024**2, 67 * 1024**2):
        unit, n = publish.chunks(size)
        assert (n - 1) * unit < size and size - (n - 1) * unit <= 128 * 1024**2


def test_metrics_affect_creative_selection():
    j = create_job("create")
    query(
        "INSERT INTO takes VALUES(?,?,?,?,?,?,?)", (j, 2, "ltx", "challenge", 80, 1, 30)
    )
    with TestClient(app) as c:
        headers = {"X-Framecraft": "1"}
        assert c.post(
            f"/api/jobs/{j}/metrics",
            json={"variant": 2, "impressions": 200, "clicks": 18, "conversions": 3},
            headers=headers,
        ).is_success
        assert angles()[0] == "challenge"
        assert (
            c.post(
                f"/api/jobs/{j}/metrics",
                json={"variant": 2, "impressions": 2, "clicks": 18, "conversions": 3},
                headers=headers,
            ).status_code
            == 400
        )


def test_local_ui_and_write_origin():
    with TestClient(app) as c:
        assert "Framecraft" in c.get("/").text
        assert c.get("/studio.js").status_code == 200
        assert c.post("/api/create", json={"brief": "test"}).status_code == 403
        assert (
            c.post(
                "/api/create",
                headers={"X-Framecraft": "1", "Origin": "https://foreign.example"},
                json={"brief": "test"},
            ).status_code
            == 403
        )
        assert (
            c.get(
                "/api/jobs/" + ("a" * 32) + "/files/%2E%2E%2Fsettings.json"
            ).status_code
            == 404
        )


def test_publish_is_idempotent(monkeypatch):
    j = create_job("create")
    (jobdir(j) / "winner.mp4").write_bytes(b"fixture")
    monkeypatch.setenv("FRAMECRAFT_META_TOKEN", "test")
    monkeypatch.setenv("FRAMECRAFT_META_AD_ACCOUNT", "42")

    class Response:
        def raise_for_status(self):
            return self

        def json(self):
            return {"id": "asset123"}

    sent = []

    def post(*args, **kwargs):
        sent.append(args)
        return Response()

    monkeypatch.setattr(publish.httpx, "post", post)
    with pytest.raises(ValueError):
        publish.deliver(j, "meta_asset", {})
    assert (
        publish.deliver(j, "meta_asset", {"consent": True})["remote_id"] == "asset123"
    )
    with pytest.raises(ValueError):
        publish.deliver(j, "meta_asset", {"consent": True})
    assert len(sent) == 1


def test_repeated_late_preview_cancels_only_matching_job(tmp_path):
    import struct

    img = cv2.imencode(".png", np.zeros((128, 128, 3), np.uint8))[1].tobytes()
    messages = iter(
        [
            json.dumps(
                {"type": "progress", "data": {"prompt_id": "abc", "value": s, "max": 8}}
            )
            if isinstance(s, int)
            else struct.pack(">II", 1, 2) + img
            for s in (5, None, 6, None)
        ]
    )

    class Socket:
        def recv(self, timeout):
            return next(messages)

    m = PreviewMonitor.__new__(PreviewMonitor)
    m.socket = Socket()
    m.folder = tmp_path
    m.step = m.maximum = m.streak = 0
    m.last_step = -1
    m.reason = ""
    m.active = False

    class Client:
        def post(self, url):
            assert url == "/api/jobs/abc/cancel"

            class Response:
                is_success = True

                def json(self):
                    return {"cancelled": True}

            return Response()

    with pytest.raises(ValueError, match="uniform exposure"):
        for _ in range(4):
            m.inspect(Client(), "abc", lambda s: None)


def test_reference_upload_then_create_from_actual_footage(tmp_path, monkeypatch):
    from time import monotonic, sleep

    src = tmp_path / "source.mp4"
    ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=256x256:rate=24",
            "-t",
            "0.75",
            "-c:v",
            "libx264",
            src,
        ]
    )
    monkeypatch.setattr(pipeline, "inventory", lambda: {"models": []})
    headers = {"X-Framecraft": "1"}
    with TestClient(app) as client:

        def wait(j):
            deadline = monotonic() + 90
            while monotonic() < deadline:
                detail = client.get("/api/jobs/" + j).json()
                if detail["status"] not in ("running", "queued"):
                    return detail
                sleep(0.2)
            pytest.fail("job did not finish")

        with src.open("rb") as f:
            upload = client.post(
                "/api/analyze",
                files={"video": ("source.mp4", f, "video/mp4")},
                headers=headers,
            )
        assert upload.status_code == 202, upload.text
        ref = upload.json()["id"]
        assert wait(ref)["result"]["analysis"]["frames"] == 18
        job = client.post(
            "/api/create",
            json={
                "brief": "Renkli oyun tahtası",
                "reference": ref,
                "use_footage": True,
                "hook": "Son hamle sende",
                "shots": 1,
                "variants": 1,
                "platform": "square",
            },
            headers=headers,
        ).json()["id"]
        result = wait(job)
        assert result["status"] == "complete", result
        assert result["result"]["winner"]["provider"] == "footage"
        audit_rows = client.get(
            f"/api/jobs/{job}/frames", params={"report": "v1/check", "limit": 18}
        ).json()
        assert len(audit_rows) == 18 and all(
            r["glyph_match"] > 0.84 for r in audit_rows
        )
        assert client.get(f"/api/jobs/{job}/files/winner.mp4").is_success


def test_oauth_rejects_foreign_state(monkeypatch):
    from framecraft import accounts

    monkeypatch.setenv("FRAMECRAFT_TIKTOK_CLIENT_ID", "local-test")
    monkeypatch.setenv("FRAMECRAFT_TIKTOK_CLIENT_SECRET", "dummy")
    link, state = accounts.start("tiktok", "http://127.0.0.1:7865")
    assert "code_challenge_method=S256" in link
    with pytest.raises(ValueError, match="state mismatch"):
        accounts.finish("tiktok", state, "wrong", "unused")
