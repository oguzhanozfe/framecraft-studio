"""Reassess existing exports together against a common creative brief."""

import os
import shutil
import time

from pydantic import BaseModel, Field

from .audit import audit
from .core import Brief, jobdir, query, read, record, save
from .director import critique
from .quality import caption_spelling


class Candidate(BaseModel):
    job: str = Field(pattern="^[a-f0-9]{32}$")
    variant: int | None = Field(None, ge=1, le=10)


class Comparison(BaseModel):
    brief: str = Field(min_length=3, max_length=3000)
    candidates: list[Candidate] = Field(min_length=2, max_length=10)


def resolve(candidate):
    source = jobdir(candidate.job)
    if candidate.variant is None:
        video = source / "input.mp4"
        if not video.is_file():
            raise ValueError("Choose a reference or an exported variant")
        return video, [], [], "import"
    folder = source / f"v{candidate.variant}"
    video = folder / "master.mp4"
    if not video.is_file():
        raise ValueError("Variant has no exported video")
    cues = read(folder / "captions.json") if (folder / "captions.json").exists() else []
    plates = [folder / f"caption-{i}.png" for i in range(len(cues))]
    takes = query(
        "SELECT provider FROM takes WHERE job=? AND variant=?",
        (candidate.job, candidate.variant),
    )
    return (
        video,
        cues,
        plates if all(p.is_file() for p in plates) else [],
        takes[0]["provider"] if takes else "export",
    )


def link_or_copy(source, dest):
    try:
        os.link(source, dest)
    except OSError:
        shutil.copyfile(source, dest)


def compare(job):
    root = jobdir(job)
    spec = Comparison.model_validate(read(root / "comparison.json"))
    ranking = []
    for i, choice in enumerate(spec.candidates, 1):
        start = time.monotonic()
        folder = root / f"v{i}"
        folder.mkdir()
        video, cues, plates, provider = resolve(choice)
        link_or_copy(video, folder / "master.mp4")
        if cues:
            save(folder / "captions.json", cues)
        for k, plate in enumerate(plates):
            link_or_copy(plate, folder / f"caption-{k}.png")
        mode = (
            "reference"
            if not cues
            else ("footage_master" if provider == "footage" else "master")
        )
        qa = audit(
            video,
            folder / "check",
            mode,
            cues,
            lambda s: record(job, "running", f"V{i} · {s}"),
            plates=plates,
        )
        record(job, "running", f"V{i} · visual review")
        review = critique(folder / "check", qa, spec.brief)
        save(folder / "critique.json", review)
        score = qa["score"]
        reasons = list(qa["blockers"])
        source_root = jobdir(choice.job)
        if cues and (source_root / "brief.json").is_file():
            brief = Brief.model_validate(read(source_root / "brief.json"))
            storyboard = source_root / f"v{choice.variant}" / "story.json"
            shots = read(storyboard)["shots"] if storyboard.is_file() else []
            spelling = caption_spelling(
                [cue["text"] for cue in cues], brief.language, brief.vocabulary(shots)
            )
            reasons.extend("Caption spelling: " + word for word in spelling)
        if not qa["passed"] and not reasons:
            reasons.append("Low technical score")
        if review["status"] == "assessed":
            score = score * 0.75 + review["brief_fit"] * 0.25
            reasons.extend(review["severe_visual_defects"])
            if review["brief_fit"] < 55:
                reasons.append("Low brief alignment")
        ranking.append(
            {
                "variant": i,
                "provider": provider,
                "angle": "comparison",
                "score": round(score, 1),
                "passed": not reasons,
                "reasons": reasons,
                "vision": review["status"],
                "source": choice.model_dump(),
                "seconds": round(time.monotonic() - start, 1),
            }
        )
        save(root / "ranking.json", ranking)
    ranked = sorted(
        ranking,
        key=lambda r: (r["passed"], r["vision"] == "assessed", r["score"]),
        reverse=True,
    )
    winner = ranked[0] if ranked[0]["passed"] else None
    if winner:
        link_or_copy(root / f"v{winner['variant']}" / "master.mp4", root / "winner.mp4")
    record_model_reviews(job, ranked)
    save(root / "result.json", {"winner": winner, "ranking": ranked})
    record(job, "complete" if winner else "needs_review", "Comparison complete")


def record_model_reviews(job, ranked):
    """Count each original take once, using its latest same-campaign visual comparison."""
    if len({row["source"]["job"] for row in ranked}) != 1:
        return
    for row in ranked:
        source = row["source"]
        if source["variant"] is not None and row["vision"] == "assessed":
            query(
                "INSERT OR REPLACE INTO model_reviews VALUES(?,?,?,?,?)",
                (
                    source["job"],
                    source["variant"],
                    job,
                    row["score"],
                    int(row["passed"]),
                ),
            )
