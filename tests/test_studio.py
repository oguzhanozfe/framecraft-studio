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


def test_revise_uses_selected_master_without_mutating_parent(monkeypatch):
    from framecraft import server
    from framecraft.core import record

    parent = create_job("create")
    root = jobdir(parent)
    spec = Brief(brief="The presenter opens a carton", variants=3, hook="Old hook")
    save(root / "brief.json", spec.model_dump())
    source = root / "v2"
    source.mkdir()
    (source / "master.mp4").write_bytes(b"previous cut")
    save(source / "story.json", {"shots": []})
    save(root / "result.json", {"winner": {"variant": 2}})
    record(parent, "complete", "Ready")
    monkeypatch.setattr(server.pool, "submit", lambda *args: None)
    with TestClient(app) as client:

        def revise(data):
            return client.post(
                f"/api/jobs/{parent}/revise", headers={"X-Framecraft": "1"}, json=data
            )

        assert revise({"changes": "New mood", "variant": 1}).status_code == 400
        assert revise({"changes": "   "}).status_code in (400, 422)
        response = revise({"changes": "Show the unpackaged product"})
        assert response.status_code == 202
        child = response.json()["id"]
        data = read(jobdir(child) / "brief.json")
        assert data["revision"] == {
            "job": parent,
            "variant": 2,
            "changes": "Show the unpackaged product",
        }
        assert data["variants"] == 3 and not data["use_footage"] and not data["hook"]
        assert client.get(f"/api/jobs/{child}").json()["revision"] == data["revision"]
    assert (source / "master.mp4").read_bytes() == b"previous cut"
    assert read(root / "brief.json") == spec.model_dump()


def test_revision_director_reads_parent_and_does_not_substitute_a_template(monkeypatch):
    from PIL import Image

    from framecraft import director
    from framecraft.core import Revision

    parent = create_job("create")
    source = jobdir(parent) / "v1"
    (source / "check").mkdir(parents=True)
    Image.new("RGB", (128, 128), "blue").save(source / "check" / "key-000001.jpg")
    save(
        source / "story.json", {"direction": "second-generation direction", "shots": []}
    )
    brief = Brief(
        brief="original",
        shots=1,
        revision=Revision(job=parent, variant=1, changes="New product shot"),
    )
    monkeypatch.setenv("FRAMECRAFT_DIRECTOR", "auto")

    def ask(schema, instruction, payload, picture=None):
        if schema is director.Observation:
            assert "changes" not in payload
            assert picture.startswith(b"\xff\xd8")
            return schema(
                visible_subjects=["blue cube"], visible_actions=[], settings=[]
            )
        assert picture is None and payload["source_observations"][
            "visible_subjects"
        ] == ["blue cube"]
        assert payload["previous_direction"] == "second-generation direction"
        assert payload["changes"] == "New product shot"
        raise RuntimeError("model offline")

    monkeypatch.setattr(director, "ask", ask)
    with pytest.raises(RuntimeError, match="model offline"):
        director.plan(brief, "reveal", {})

    scenes = [
        "The rectangular carton sits prominently on the desk under a reading lamp.",
        "An unpackaged round blue speaker rotates on the wood desk in side light.",
    ]

    def draft(schema, instruction, payload, picture=None):
        if schema is director.CopyDeck:
            assert payload["language"] == "tr" and payload["captions"] == ["Detay"]
            return schema(captions=["Yakından bak"])
        if schema is director.Observation:
            return schema(
                visible_subjects=["blue carton"], visible_actions=[], settings=[]
            )
        if schema is director.ShotEvidence:
            assert payload == {"visible_scene": scenes[1]}
            return schema(
                subject_detail="Round blue speaker with a grille",
                checks=["Round speaker is visible"],
            )
        if schema is director.PlanIssues:
            return schema(
                issues=["The item is still inside its container"]
                if "carton" in payload["scenes"][0]["visual"]
                else []
            )
        revision = "corrections" in payload
        index = int(revision)
        if revision:
            assert "container" in payload["corrections"][0]
        return schema(
            direction="Present a blue speaker",
            final_checks=["unboxed speaker"],
            shots=[
                {
                    "visual": scenes[index],
                    "subject_detail": "Round blue speaker with a grille",
                    "checks": ["visible subject"],
                    "caption": "Detay",
                }
            ],
        )

    monkeypatch.setattr(director, "ask", draft)
    story = director.plan(brief, "reveal", {})
    assert story["shots"][0]["visual"] == scenes[1] and story["planning_feedback"]
    assert story["shots"][0]["caption"] == "Yakından bak"


def test_rejected_plans_do_not_render_or_hide_the_other_variant(monkeypatch):
    from framecraft.core import Revision

    j = create_job("create")
    root = jobdir(j)
    save(
        root / "brief.json",
        Brief(
            brief="previous",
            variants=2,
            revision=Revision(job="f" * 32, variant=1, changes="change the prop"),
        ).model_dump(),
    )
    monkeypatch.setattr(
        pipeline, "inventory", lambda: {"models": [{"id": "h3", "available": True}]}
    )

    def rejected(*args):
        raise ValueError("Plan contradicts requested prop")

    monkeypatch.setattr(pipeline, "plan", rejected)
    monkeypatch.setattr(
        pipeline, "generate", lambda *args: pytest.fail("rejected plan rendered")
    )
    pipeline.make(j)
    ranking = read(root / "result.json")["ranking"]
    assert len(ranking) == 2 and all(not row["passed"] for row in ranking)
    assert all((root / f"v{i}" / "planning-error.json").is_file() for i in (1, 2))


def test_fresh_model_generation_requires_a_director(monkeypatch):
    import httpx

    from framecraft import director

    monkeypatch.setenv("FRAMECRAFT_DIRECTOR", "auto")

    def offline(*args):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(director, "ask", offline)
    with pytest.raises(ValueError, match="director connection"):
        director.plan(Brief(brief="show a racing bike"), "demo", {})
    assert director.plan(
        Brief(brief="cut the source footage", use_footage=True), "demo", {}
    )["director"].startswith("local storyboard")


@pytest.mark.parametrize(
    "results,expected_status",
    [
        ([(1, "unclear"), (2, "met")], "assessed"),
        ([(1, "met")], "assessed"),
        ([(0, "met"), (1, "met")], "assessed"),
        ([(1, "met"), (1, "met")], "unavailable"),
    ],
)
def test_revision_checks_require_complete_unambiguous_evidence(
    tmp_path, monkeypatch, results, expected_status
):
    from PIL import Image

    from framecraft import director

    Image.new("RGB", (100, 100), "blue").save(tmp_path / "key-000000.jpg")
    monkeypatch.setenv("FRAMECRAFT_DIRECTOR", "auto")

    def ask(schema, instruction, payload, picture):
        if schema is director.Observation:
            assert "required_checks" not in payload
            return schema(visible_subjects=["square"], settings=[], visible_actions=[])
        if schema is director.Verdict:
            assert picture and "blind_observations" in payload
            if expected_status == "unavailable":
                raise ValueError("Vision was interrupted")
            if payload["condition"] == "show a sphere":
                return schema(
                    result="met", observation="Visible rounded object in frame 0"
                )
            assert payload["condition"] == "blue"
            return schema(
                result="unclear", observation="Lighting prevents a color reading"
            )
        assert len(payload["required_checks"]) == 2
        return schema(
            subject_evidence="visible",
            action_evidence="visible",
            style="",
            hook="",
            audience="",
            beats=[],
            evidence=[],
            improvements=[],
            severe_visual_defects=[],
            alignment="strong",
            evidence_by_check=[
                {"check_id": i, "result": r, "observation": "Frame 0 shows a square"}
                for i, r in results
            ],
        )

    monkeypatch.setattr(director, "ask", ask)
    result = director.critique(
        tmp_path, {"frequent_text": [], "shots": []}, checks=["show a sphere", "blue"]
    )
    assert result["status"] == expected_status
    if expected_status == "assessed":
        assert result["brief_fit"] <= 40 and len(result["failed_checks"]) == 1
        assert [e["check_id"] for e in result["evidence_by_check"]] == [1, 2]


@pytest.mark.parametrize(
    "final_status", ["assessed", "unavailable", "shot_unavailable", "recovered"]
)
def test_model_retry_and_final_gate(monkeypatch, final_status):
    from framecraft.core import Revision

    job = create_job("create")
    root = jobdir(job)
    brief = Brief(
        brief="show the carton",
        variants=1,
        shots=1,
        revision=Revision(job="a" * 32, variant=1, changes="show the bare item"),
    )
    save(root / "brief.json", brief.model_dump())
    shot = {"visual": "First model idea", "caption": "Detay", "checks": ["round item"]}
    story = {
        "angle": "detail",
        "shots": [shot],
        "direction": "unpackaged product",
        "final_checks": ["unpackaged item"],
    }
    monkeypatch.setattr(pipeline, "plan", lambda *args: story)
    if final_status == "recovered":
        parent = create_job("create")
        variant = jobdir(parent) / "v1"
        prior = variant / "shot-1-1"
        (prior / "audit").mkdir(parents=True)
        (prior / "raw.mp4").write_bytes(b"model-generated source")
        save(prior / "review.json", {"status": "unavailable"})
        save(prior / "audit" / "report.json", {"passed": True})
        save(variant / "story.json", story)
        save(
            variant / "attempts.json",
            [
                {
                    "shot": 1,
                    "attempt": 1,
                    "provider": "h3",
                    "prompt": "original render prompt",
                    "seed": 12,
                }
            ],
        )
        save(root / "retry-source.json", {"job": parent})
        monkeypatch.setattr(pipeline, "copyedit", lambda shots, language: None)
    monkeypatch.setattr(
        pipeline, "inventory", lambda: {"models": [{"id": "h3", "available": True}]}
    )
    prompts = []

    def generate(provider, prompt, seed, platform, output, progress, lora):
        prompts.append(prompt)
        output.write_bytes(b"renderer output")
        return output

    monkeypatch.setattr(pipeline, "generate", generate)

    def rewrite(shot, visual, findings):
        assert "carton still visible" in findings and shot["checks"] == ["round item"]
        return "A different model-directed shot"

    monkeypatch.setattr(pipeline, "rewrite_shot", rewrite)
    monkeypatch.setattr(
        pipeline,
        "audit",
        lambda *args, **kwargs: {
            "score": 99,
            "passed": True,
            "blockers": [],
            "frames": 3,
        },
    )

    def render(clips, captions, shape, folder, repair, preserve_source):
        path = folder / "master.mp4"
        path.write_bytes(clips[0].read_bytes())
        return path, []

    monkeypatch.setattr(pipeline, "render", render)

    def critique(folder, report, direction, *, checks):
        if final_status == "shot_unavailable":
            return {"status": "unavailable", "reason": "offline"}
        if checks == story["final_checks"]:
            assert (
                "show the bare item" in direction and "show the carton" not in direction
            )
            return {
                "status": "assessed" if final_status == "recovered" else final_status,
                "brief_fit": 95,
                "severe_visual_defects": [],
            }
        assert checks == shot["checks"]
        passed = len(prompts) > 1 or final_status == "recovered"
        return {
            "status": "assessed",
            "brief_fit": 95 if passed else 40,
            "severe_visual_defects": [],
            "failed_checks": [] if passed else ["carton still visible"],
        }

    monkeypatch.setattr(pipeline, "critique", critique)
    pipeline.make(job)
    if final_status == "recovered":
        assert (
            prompts == []
            and (root / "winner.mp4").read_bytes() == b"model-generated source"
        )
        assert read(root / "v1" / "attempts.json")[0]["source_job"] == parent
        return
    if final_status == "shot_unavailable":
        assert len(prompts) == 1 and not (root / "winner.mp4").exists()
        assert "offline" in read(root / "result.json")["ranking"][0]["reasons"][0]
        return
    assert len(prompts) == 2 and prompts[1].startswith(
        "A different model-directed shot"
    )
    assert read(root / "v1" / "attempts.json")[0]["visual_findings"] == [
        "carton still visible"
    ]
    assert (root / "winner.mp4").exists() == (final_status == "assessed")


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


def test_director_retries_unsupported_local_options_with_schema(monkeypatch):
    import httpx

    from framecraft import director

    requests = []

    def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        if "reasoning_effort" in body:
            return httpx.Response(422)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"visual":"A round blue ball"}'}}]
            },
        )

    client = httpx.Client
    monkeypatch.setenv("FRAMECRAFT_LLM_MODEL", "example")
    monkeypatch.setenv("FRAMECRAFT_LLM_URL", "http://127.0.0.1:1234/v1")
    monkeypatch.setattr(
        director.httpx,
        "Client",
        lambda **kwargs: client(**kwargs, transport=httpx.MockTransport(handle)),
    )
    assert (
        director.ask(director.NewVisual, "Write a prompt", {}).visual
        == "A round blue ball"
    )
    assert (
        requests[0]["reasoning_effort"] == "none"
        and "chat_template_kwargs" not in requests[1]
    )
    assert requests[0]["response_format"] == requests[1]["response_format"]


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


def test_compare_prioritizes_evidence_and_preserves_originals(monkeypatch):
    from framecraft import compare as module

    refs = [create_job("reference") for _ in range(2)]
    for i, r in enumerate(refs):
        (jobdir(r) / "input.mp4").write_bytes(bytes([i]))
    job = create_job("compare")
    save(
        jobdir(job) / "comparison.json",
        {"brief": "Puzzle board", "candidates": [{"job": r} for r in refs]},
    )
    monkeypatch.setattr(
        module,
        "audit",
        lambda *args, **kwargs: {"score": 98, "passed": True, "blockers": []},
    )
    reviews = iter(
        [
            {"status": "assessed", "brief_fit": 40, "severe_visual_defects": []},
            {"status": "assessed", "brief_fit": 95, "severe_visual_defects": []},
        ]
    )
    monkeypatch.setattr(module, "critique", lambda *args: next(reviews))
    module.compare(job)
    result = read(jobdir(job) / "result.json")
    assert result["winner"]["variant"] == 2
    assert "Low brief alignment" in result["ranking"][1]["reasons"]
    assert (jobdir(job) / "winner.mp4").read_bytes() == bytes([1])
    assert (jobdir(refs[0]) / "input.mp4").read_bytes() == bytes([0])


def test_review_grounds_alignment_in_visible_subject(tmp_path, monkeypatch):
    from PIL import Image

    from framecraft import director

    monkeypatch.setenv("FRAMECRAFT_DIRECTOR", "auto")
    Image.new("RGB", (100, 100), "blue").save(tmp_path / "key-000000.jpg")
    calls = []

    def ask(schema, prompt, payload, picture):
        calls.append(payload)
        if schema is director.Observation:
            return schema(
                visible_subjects=["A blue square"],
                settings=["plain background"],
                visible_actions=["none"],
            )
        return schema(
            style="minimal",
            hook="still",
            audience="unknown",
            beats=[],
            evidence=[],
            improvements=[],
            severe_visual_defects=[],
            alignment="strong",
            subject_evidence="missing",
            action_evidence="missing",
        )

    monkeypatch.setattr(director, "ask", ask)
    result = director.critique(
        tmp_path, {"frequent_text": [], "shots": []}, "Show a running horse"
    )
    assert "brief" not in calls[0] and "independent_observations" in calls[1]
    assert result["brief_fit"] < 55


def test_same_campaign_review_history_does_not_double_count():
    from framecraft.compare import record_model_reviews

    job = create_job("create")
    ranked = [
        {
            "source": {"job": job, "variant": 1},
            "vision": "assessed",
            "passed": True,
            "score": 90,
        }
    ]
    for run in ("one", "two"):
        record_model_reviews(run, ranked)
    rows = query("SELECT * FROM model_reviews WHERE job=?", (job,))
    assert len(rows) == 1 and rows[0]["comparison"] == "two"
