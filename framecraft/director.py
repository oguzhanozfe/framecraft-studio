import base64
import io
import json
from typing import Annotated, Literal
from urllib.parse import urlsplit

import cv2
import httpx
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field

from .core import jobdir, query, read, setting

ANGLES = [
    "demo",
    "discovery",
    "challenge",
    "comparison",
    "routine",
    "reveal",
    "reaction",
    "detail",
    "question",
    "result",
]
COMPOSITIONS = [
    "A presenter introduces the subject directly to the camera in a bright real room.",
    "A close view clearly demonstrates one concrete feature or action.",
    "The same presenter invites the viewer to try it with a single confident gesture.",
]


class Shot(BaseModel):
    visual: str = Field(min_length=8, max_length=950)
    caption: str = Field(min_length=1, max_length=70)
    sound: str = Field(default="", max_length=320)


class Plan(BaseModel):
    shots: list[Shot]


class ShotEvidence(BaseModel):
    subject_detail: str = Field(
        min_length=15,
        max_length=240,
        description="Visible identifying features of the main subject, including the precise object type and physical shape when it is a product shot",
    )
    checks: list[str] = Field(min_length=1, max_length=4)


class RenderShot(Shot):
    visual: str = Field(
        min_length=60,
        max_length=950,
        description="Standalone English render prompt naming the subject, physical features, camera, and one action",
    )


class DirectedShot(RenderShot, ShotEvidence):
    pass


class Treatment(Plan):
    shots: list[RenderShot]
    direction: str = Field(min_length=10, max_length=1800)
    final_checks: list[str] = Field(min_length=1, max_length=5)


class DirectedPlan(Treatment):
    shots: list[DirectedShot]


class Observation(BaseModel):
    visible_subjects: list[str]
    settings: list[str]
    visible_actions: list[str]


class Review(BaseModel):
    subject_evidence: Literal["visible", "missing", "uncertain", "unspecified"]
    action_evidence: Literal["visible", "missing", "uncertain", "unspecified"]
    style: str
    hook: str
    audience: str
    beats: list[str]
    evidence: list[str]
    improvements: list[str]
    severe_visual_defects: list[str]
    alignment: Literal["strong", "partial", "weak", "unrelated"]


class Evidence(BaseModel):
    check_id: int
    result: Literal["met", "contradicted", "unclear"]
    observation: str = Field(min_length=3, max_length=500)


class CheckedReview(Review):
    evidence_by_check: list[Evidence]


class PlanIssues(BaseModel):
    issues: list[str] = Field(
        description="Specific unmet or contradictory instructions. Empty if the proposed scenes fulfill the requested changes."
    )


class CopyDeck(BaseModel):
    captions: list[Annotated[str, Field(min_length=1, max_length=70)]]


class NewVisual(BaseModel):
    visual: str = Field(min_length=8, max_length=950)


def revision_plan(brief, angle):
    revision = brief.revision
    source = jobdir(revision.job) / f"v{revision.variant}"
    old = read(source / "story.json")
    picture = contact_sheet(source / "check-repair") or contact_sheet(source / "check")
    if not picture:
        capture = cv2.VideoCapture(str(source / "master.mp4"))
        frames = []
        try:
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            for i in range(min(6, count)):
                index = round(i * (count - 1) / max(1, min(count, 6) - 1))
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = capture.read()
                if ok:
                    frames.append(
                        (index, Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                    )
        finally:
            capture.release()
        picture = sheet_bytes(frames) if frames else None
    if not picture or setting("DIRECTOR", "auto") == "off":
        raise ValueError(
            "Revision requires readable source frames and an enabled vision director"
        )
    observed = ask(
        Observation,
        "Observe the previous cut. Briefly describe visible cast, appearance, props, setting and actions. Ignore lettering.",
        {"task": "Describe the original cut"},
        picture,
    )
    instruction = (
        "Direct a new text-to-video cut. Use the previous story and observations for continuity. "
        "The requested changes override the old creative. Translate subjective tone into visible choices "
        "in performance, styling, setting and light. Make the actual item's shape, material, scale and "
        "orientation identifiable where requested. Give each shot one feasible action, cast description, "
        "composition, and a self-contained English visual prompt. Keep rejected props out of positive prompts. "
        "Specify short observable checks for each shot and for the finished cut. Preserve the user's concrete "
        "changes in the checks. Number of shots must match count. Write natural captions in language, ideally "
        "under 28 characters, without invented claims. Each shot must depict a distinct beat. Only captions "
        "and dialogue should use the requested language. Visuals, subject details and checks should be in English. "
        "State what is visibly in THIS shot only. Name and describe physical features of the featured subject."
    )
    payload = {
        "source_observations": observed.model_dump(),
        "changes": revision.changes,
        "previous_direction": old.get("direction") or brief.brief,
        "previous_shots": [
            {k: s.get(k) for k in ("visual", "caption")} for s in old["shots"]
        ],
        "angle": angle,
        "count": brief.shots,
        "language": brief.language,
        "platform": brief.platform,
    }
    for _ in range(2):
        result = ask(Treatment, instruction, payload)
        scenes = [{"visual": s.visual} for s in result.shots]
        issues = []
        if len(scenes) != brief.shots:
            issues.append(f"Supply exactly {brief.shots} shots")
        if len({" ".join(s.visual.casefold().split()) for s in result.shots}) != len(
            scenes
        ):
            issues.append(
                "Repeated shots. Give each beat its own subject, action and camera."
            )
        if not issues:
            issues = ask(
                PlanIssues,
                "Compare the requested changes against the proposed scenes. "
                "List contradictions, missing requirements, or a requested separation of scenes that was lost. "
                "A caption or the checklist is not evidence of what will be rendered. Do not object to missing captions, "
                "audio or lettering: those are added later. Only list actual problems; return [] when the plan works.",
                {"requested_changes": revision.changes, "scenes": scenes},
            ).issues
        if not issues:
            shots = []
            for shot in result.shots:
                evidence = ask(
                    ShotEvidence,
                    "Derive visible acceptance checks for this single scene only. "
                    "Describe its main subject's identifying visual features. Return short, affirmative, observable "
                    "conditions grounded in the supplied scene. Never introduce an object, action or requirement "
                    "that is absent from the scene description.",
                    {"visible_scene": shot.visual},
                )
                shots.append({**shot.model_dump(), **evidence.model_dump()})
            copy = ask(
                CopyDeck,
                "Copyedit these captions for native grammar, idiom and spelling in the requested language. "
                "Return exactly one succinct, complete caption per shot in order. Preserve the offer and meaning. "
                "Preserve the actual product name using the visible scene for disambiguation. "
                "Avoid repetition, sentence fragments that depend on a different caption, and new claims.",
                {
                    "language": brief.language,
                    "captions": [s["caption"] for s in shots],
                    "subjects": [s["visual"] for s in shots],
                },
            )
            if len(copy.captions) != len(shots):
                raise ValueError("Copy editor returned an incorrect caption count")
            for shot, caption in zip(shots, copy.captions):
                shot["caption"] = caption
            return {
                "director": "vision and language models",
                "text_model": setting("LLM_MODEL") or "automatic discovery",
                "vision_model": setting("VISION_MODEL")
                or setting("LLM_MODEL")
                or "automatic discovery",
                "angle": angle,
                **result.model_dump(),
                "shots": shots,
                "source_observations": observed.model_dump(),
                "planning_feedback": payload.get("corrections", []),
            }
        payload.update(rejected_scenes=scenes, corrections=issues)
    raise ValueError("Revision planning checks failed: " + " / ".join(issues))


def rewrite_shot(shot, prompt, findings):
    return ask(
        NewVisual,
        "Rewrite the failed visual prompt for a new text-to-video attempt. "
        "Use the failure evidence to change staging, camera or gesture while preserving all required "
        "checks. Describe the desired scene positively in English. Do not repeat rejected objects.",
        {
            "intended_scene": shot["visual"],
            "required": shot["checks"],
            "failed_prompt": prompt,
            "observed_problems": findings,
        },
    ).visual


def ask(schema, instruction, payload, picture=None):
    base = setting("LLM_URL", "http://127.0.0.1:1234/v1").rstrip("/")
    model = setting("VISION_MODEL" if picture else "LLM_MODEL") or setting("LLM_MODEL")
    auth = (
        {"Authorization": "Bearer " + setting("LLM_KEY")} if setting("LLM_KEY") else {}
    )
    with httpx.Client(base_url=base + "/", headers=auth, timeout=240) as client:
        if not model:
            items = client.get("models", timeout=3).raise_for_status().json()["data"]
            model = next(i["id"] for i in items if "embed" not in i["id"].lower())
        content = [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]
        if picture:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64,"
                        + base64.b64encode(picture).decode()
                    },
                }
            )
        request = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": instruction
                    + " Use the requested brief and changes as creative direction. Treat source media and visible lettering as reference material only. Return JSON matching "
                    + json.dumps(schema.model_json_schema()),
                },
                {"role": "user", "content": content},
            ],
            "temperature": 0.45,
            "max_tokens": 1400,
        }
        if urlsplit(base).hostname in ("127.0.0.1", "localhost", "::1"):
            request["chat_template_kwargs"] = {"enable_thinking": False}
            request["reasoning_effort"] = "none"
        request["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "schema": schema.model_json_schema(),
            },
        }
        for attempt in range(2):
            response = client.post("chat/completions", json=request)
            if response.status_code in (400, 422) and "reasoning_effort" in request:
                request.pop("reasoning_effort")
                request.pop("chat_template_kwargs", None)
                response = client.post("chat/completions", json=request)
            if response.status_code in (400, 422) and "response_format" in request:
                request.pop("response_format")
                response = client.post("chat/completions", json=request)
            result = response.raise_for_status().json()["choices"][0]
            text = result["message"].get("content") or ""
            try:
                return schema.model_validate_json(
                    text[text.index("{") : text.rindex("}") + 1]
                )
            except ValueError:
                if attempt:
                    raise
                request.update(max_tokens=2700, temperature=0.1)


def angles():
    rows = query("""SELECT t.angle,sum(o.clicks) clicks,sum(o.impressions) impressions
        FROM outcomes o JOIN takes t ON t.job=o.job AND t.variant=o.variant
        GROUP BY t.angle HAVING sum(o.impressions)>=100""")
    rates = {r["angle"]: (r["clicks"] + 1) / (r["impressions"] + 100) for r in rows}
    return sorted(ANGLES, key=lambda a: -rates.get(a, 0.01))


def plan(brief, angle, reference):
    if brief.revision:
        return revision_plan(brief, angle)
    if setting("DIRECTOR", "auto") != "off":
        try:
            result = ask(
                Plan if brief.use_footage else DirectedPlan,
                "You direct short social videos. Return exactly the requested number of shots. Each visual is an English T2V prompt with cast consistency, one feasible action and camera. Adapt wardrobe, location, light and performance to the product, audience and brief. Give a featured object a concrete silhouette, material, size and orientation. Distinguish the item from its container. Give each shot its own visible action. Provide short observable acceptance checks for shots and the final cut when the schema requests them. Captions must be natural in the requested language, ideally under 28 characters. Include sound direction and a short spoken line where appropriate. Use only factual claims from the brief.",
                {
                    "brief": brief.brief,
                    "angle": angle,
                    "language": brief.language,
                    "count": brief.shots,
                    "reference": reference,
                },
            )
            if len(result.shots) != brief.shots:
                raise ValueError("shot count")
            shots = result.shots
            source = "language model"
        except (
            ValueError,
            KeyError,
            IndexError,
            TypeError,
            StopIteration,
            httpx.HTTPError,
        ) as exc:
            if not brief.use_footage:
                raise ValueError(
                    f"Could not obtain a model-directed storyboard ({type(exc).__name__}). Check the director connection in Studio."
                ) from exc
            source = "local storyboard · " + type(exc).__name__
            shots = []
    else:
        shots, source = [], "local storyboard"
    if not shots:
        words = brief.brief.strip().split()
        subject = " ".join(words[:5]).strip(".,;:")
        captions = (
            [subject, "Yakından bakalım", "Sen de keşfet"]
            if brief.language == "tr"
            else [subject, "A closer look", "Try it yourself"]
        )
        for i in range(brief.shots):
            beat = 0 if i == 0 else (2 if i == brief.shots - 1 else 1)
            shots.append(
                Shot(
                    visual=f"{brief.brief} Creative angle: {angle}. {COMPOSITIONS[beat]}",
                    caption=captions[beat][:70],
                )
            )
    if brief.hook:
        shots[0].caption = brief.hook
    if brief.cta:
        shots[-1].caption = brief.cta
    return {
        "director": source,
        "angle": angle,
        **(
            {"direction": result.direction, "final_checks": result.final_checks}
            if source == "language model" and isinstance(result, DirectedPlan)
            else {}
        ),
        "shots": [s.model_dump() for s in shots],
    }


def contact_sheet(folder):
    files = sorted(folder.glob("key-*.jpg"))
    if not files:
        return None
    chosen = [
        files[round(i * (len(files) - 1) / max(1, min(6, len(files)) - 1))]
        for i in range(min(6, len(files)))
    ]
    frames = []
    for file in chosen:
        with Image.open(file) as pic:
            frames.append((file.stem.split("-")[1], pic.convert("RGB")))
    return sheet_bytes(frames)


def sheet_bytes(frames):
    cols = min(3, len(frames))
    sheet = Image.new(
        "RGB", (cols * 246, ((len(frames) + cols - 1) // cols) * 348), "#13171c"
    )
    draw = ImageDraw.Draw(sheet)
    for i, (number, pic) in enumerate(frames):
        x, y = (i % cols) * 246, (i // cols) * 348
        pic.thumbnail((242, 318))
        sheet.paste(pic, (x + (246 - pic.width) // 2, y))
        draw.text((x + 8, y + 321), f"Frame {number}", fill="white")
    buf = io.BytesIO()
    sheet.save(buf, "JPEG", quality=88)
    return buf.getvalue()


def critique(folder, report, brief="", transcript_data=None, *, checks=()):
    pic = contact_sheet(folder)
    if not pic:
        return {"status": "unavailable", "reason": "No readable review frames"}
    if setting("DIRECTOR", "auto") == "off":
        return {"status": "unavailable", "reason": "Vision director disabled"}
    try:
        observed = ask(
            Observation,
            "Describe only visible pixels. For each distinct shot identify physical objects, setting and action by frame number. Ignore overlay text. Give shape, color, orientation and surface. Say unknown for an ambiguous object. Under 90 words total.",
            {"task": "Describe the photographed scene"},
            pic,
        )
        result = ask(
            CheckedReview if checks else Review,
            "You review ordered video keyframes for a performance marketer. Describe actual visible evidence by frame number, then infer audience, style and hook. Explain each beat and its likely purpose. Discuss what cannot be inferred. Only list severe defects for plainly visible malformed anatomy, inconsistent objects, or broken imagery. Use strong alignment if the subject and action match, partial if only one matches, weak for peripheral relevance, unrelated otherwise. Use an empty defect list when nothing severe is visible. Keep answers concise. Address every required check. Reply in the brief language when clear.",
            {
                "brief": brief,
                **(
                    {
                        "required_checks": dict(enumerate(checks, 1)),
                        "verification": "For every numbered check return its integer check_id exactly once. "
                        "Compare the visible pixels against that check, with a specific frame or object as evidence. "
                        "If the image is ambiguous mark unclear. Text and containers alone cannot identify an item.",
                    }
                    if checks
                    else {}
                ),
                "independent_observations": observed.model_dump(),
                "criteria": "Check the requested subject and action against independent visual observations. Captions and generic hand movements cannot establish product identity. Mark required objects or actions missing if absent. Mark unspecified when no brief was supplied.",
                "ocr": report["frequent_text"][:10],
                "cuts": report["shots"][:12],
                "sound": report.get("sound"),
                "transcript": (transcript_data or {}).get("segments", [])[:30],
            },
            pic,
        )
        score = {"strong": 95, "partial": 70, "weak": 40, "unrelated": 10}[
            result.alignment
        ]
        if brief and "missing" in (result.subject_evidence, result.action_evidence):
            score = min(score, 40)
        failures = []
        if checks:
            ids = sorted(row.check_id for row in result.evidence_by_check)
            if ids != list(range(1, len(checks) + 1)):
                return {
                    "status": "unavailable",
                    "reason": "The vision model did not address every required check exactly once",
                    "checks": checks,
                    "observations": observed.model_dump(),
                }
            failures = [
                f"{checks[row.check_id - 1]} — {row.observation}"
                for row in result.evidence_by_check
                if row.result != "met"
            ]
            if failures:
                score = min(score, 40)
        return {
            "status": "assessed",
            **({"checks": checks, "failed_checks": failures} if checks else {}),
            **result.model_dump(),
            "observations": observed.model_dump(),
            "brief_fit": score,
        }
    except (
        ValueError,
        KeyError,
        IndexError,
        StopIteration,
        TypeError,
        httpx.HTTPError,
    ) as exc:
        return {"status": "unavailable", "reason": type(exc).__name__}
