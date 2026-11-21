import secrets
import shutil
import time

from .audit import audit
from .core import Brief, jobdir, query, read, record, save, setting
from .director import angles, critique, plan, rewrite_shot
from .edit import render
from .footage import cut
from .providers import generate, inventory
from .quality import transcript

FIXES = {
    "unplanned_text": "Plain unprinted props, blank background, no signage, keep all screens turned away.",
    "black": "Bright daylight and a clearly exposed subject from the first frame.",
    "frozen": "One continuous, visible, natural action from the opening to the end.",
    "flash": "Steady natural lighting, fixed exposure and a single continuous take.",
    "soft": "Keep the subject in sharp focus and include tactile detail.",
    "white": "Balanced daylight, clear outlines and normal exposure.",
}


def analyze_job(job):
    folder = jobdir(job)
    report = audit(
        folder / "input.mp4",
        folder / "analysis",
        progress=lambda s: record(job, "running", s),
    )
    record(job, "running", "Reading scene keyframes")
    speech = transcript(folder / "input.mp4")
    review = critique(folder / "analysis", report, transcript_data=speech)
    save(folder / "analysis" / "critique.json", review)
    save(
        folder / "result.json",
        {"analysis": report, "critique": review, "transcript": speech},
    )
    record(job, "complete", f"{report['frames']} frames analyzed")


def make(job):
    root = jobdir(job)
    brief = Brief.model_validate(read(root / "brief.json"))
    progress = lambda s: record(job, "running", s)
    progress("Checking model files and hardware")
    inv = inventory()
    save(root / "inventory.json", inv)
    available = [
        m["id"]
        for m in inv["models"]
        if m["available"] and (brief.provider == "auto" or m["id"] == brief.provider)
    ]
    if brief.use_footage:
        if not brief.reference:
            raise ValueError("Choose a reference for source footage")
        available = ["footage"]
    if not available:
        raise RuntimeError("No compatible renderer. See Models and Settings.")
    reference = {}
    if brief.reference and not brief.revision:
        ref = read(jobdir(brief.reference) / "result.json")
        reference = {
            "style": ref.get("critique", {}),
            "shots": ref["analysis"]["shots"][:12],
            "text": ref["analysis"]["frequent_text"][:8],
        }
    ranking = []
    angle_order = angles()
    lora = setting("H3_LORA") or None
    triggers = ""
    if setting("AUTO_LORA") == "on" and "h3" in available:
        from .catalog import discover, install_lora

        try:
            topic = brief.revision.changes if brief.revision else brief.brief
            entries = discover(" ".join(topic.split()[:4]))
            eligible = next(
                (r for r in entries if "Image" in (r["commercial_use"] or [])), None
            )
            if eligible:
                lora = install_lora(eligible["version"]).name
                triggers = ", ".join(eligible["trigger_words"])[:300]
                reference["adapter_prompt_examples"] = eligible["examples"]
                save(root / "adapter.json", eligible)
        except Exception as exc:
            save(
                root / "adapter.json",
                {"status": "catalog unavailable", "reason": type(exc).__name__},
            )
    for vi in range(brief.variants):
        start = time.monotonic()
        folder = root / f"v{vi + 1}"
        folder.mkdir()
        progress(f"Variant {vi + 1}/{brief.variants} · storyboard")
        story = plan(brief, angle_order[vi % len(angle_order)], reference)
        save(folder / "story.json", story)
        direction_input = brief.revision.changes if brief.revision else brief.brief
        chosen = []
        scores = []
        events = []
        providers = []
        reasons = []
        for si, shot in enumerate(story["shots"]):
            visual = shot["visual"]
            checks = list(shot.get("checks", []))
            if shot.get("subject_detail"):
                checks.append(shot["subject_detail"])
                visual += " " + shot["subject_detail"]
            hint = "Clean frame. No visible writing, logos, subtitles or watermark. Natural movement, one continuous shot."
            seed = secrets.randbelow(2**31 - 1)
            for attempt in range(2):
                path = folder / f"shot-{si + 1}-{attempt + 1}"
                path.mkdir()
                renderer = (
                    available[(vi + attempt) % len(available)]
                    if attempt or vi
                    else available[0]
                )
                # Prefer the same renderer for visual consistency when repairing image content.
                if attempt and events and "error" not in events[-1]:
                    renderer = events[-1]["provider"]
                    if checks:
                        progress("Director revising the rejected shot")
                        try:
                            visual = rewrite_shot(
                                {**shot, "checks": checks},
                                direction_input,
                                visual,
                                events[-1]["blockers"],
                            )
                        except Exception as exc:
                            events.append(
                                {
                                    "shot": si + 1,
                                    "attempt": attempt + 1,
                                    "provider": renderer,
                                    "error": f"Could not obtain revised prompt ({type(exc).__name__})",
                                }
                            )
                            save(folder / "attempts.json", events)
                            continue
                prompt = " ".join(
                    [
                        visual,
                        shot.get("sound", ""),
                        triggers if renderer == "h3" else "",
                        hint,
                    ]
                ).strip()
                event = {
                    "shot": si + 1,
                    "attempt": attempt + 1,
                    "provider": renderer,
                    "seed": seed,
                    "prompt": prompt,
                }
                events.append(event)
                save(folder / "attempts.json", events)
                try:
                    progress(f"Variant {vi + 1} · shot {si + 1} · {renderer}")
                    if renderer == "footage":
                        refdir = jobdir(brief.reference)
                        clip, source_time = cut(
                            refdir / "input.mp4",
                            refdir / "analysis",
                            si,
                            len(story["shots"]),
                            vi,
                            attempt,
                            path / "raw.mp4",
                        )
                        event["source_start"] = source_time
                    else:
                        clip = generate(
                            renderer,
                            prompt,
                            seed,
                            brief.platform,
                            path / "raw.mp4",
                            progress,
                            lora,
                        )
                    qa = audit(
                        clip,
                        path / "audit",
                        "footage" if brief.use_footage else "raw",
                        progress=progress,
                    )
                    event.update(
                        score=qa["score"], blockers=qa["blockers"], frames=qa["frames"]
                    )
                    if qa["passed"] and (
                        checks or setting("SHOT_VISION", "on") == "on"
                    ):
                        progress("Reviewing shot composition and continuity")
                        review = critique(
                            path / "audit", qa, shot["visual"], checks=checks
                        )
                        save(path / "review.json", review)
                        if checks and review["status"] != "assessed":
                            qa["passed"] = False
                            qa["blockers"] = [
                                "Visual checks unavailable: "
                                + review.get("reason", "unknown")
                            ]
                        elif review["status"] == "assessed" and (
                            review["severe_visual_defects"] or review["brief_fit"] < 50
                        ):
                            qa["passed"] = False
                            qa["blockers"] = review.get("failed_checks", []) + review[
                                "severe_visual_defects"
                            ] or ["Subject or action does not match the shot"]
                        if not qa["passed"]:
                            event["visual_findings"] = qa["blockers"]
                    if qa["passed"]:
                        chosen.append(clip)
                        providers.append(renderer)
                        scores.append(qa["score"])
                        break
                    findings = qa["blockers"] or list(qa["flag_counts"])
                    event["blockers"] = findings or ["Technical score under 75"]
                    if not checks:
                        hint = " ".join(
                            FIXES.get(k, "Correct this visible defect: " + k)
                            for k in findings
                        )
                except Exception as exc:
                    event["error"] = str(exc)[:700]
                    hint = "Use a simple clean composition."
                finally:
                    save(folder / "attempts.json", events)
                seed = secrets.randbelow(2**31 - 1)
            else:
                problems = events[-1].get("blockers") or [
                    events[-1].get("error", "technical score")
                ]
                reasons.append(
                    f"Shot {si + 1} failed both attempts: " + " / ".join(problems)
                )
                break
        if len(chosen) == len(story["shots"]):
            captions = [s["caption"] for s in story["shots"]]
            for repair in (False, True):
                progress(
                    "Typesetting captions"
                    if not repair
                    else "Repairing caption contrast and size"
                )
                master, cues = render(
                    chosen,
                    captions,
                    brief.platform,
                    folder,
                    repair,
                    preserve_source=brief.use_footage,
                )
                save(folder / "captions.json", cues)
                qa = audit(
                    master,
                    folder / ("check-repair" if repair else "check"),
                    "footage_master" if brief.use_footage else "master",
                    cues,
                    progress,
                    plates=[folder / f"caption-{i}.png" for i in range(len(cues))],
                )
                if not any(k.startswith("caption_") for k in qa["blockers"]) or repair:
                    break
            review_dir = folder / ("check-repair" if repair else "check")
            progress("Reviewing the final cut against the brief")
            direction = (
                story["direction"] + "\nRequested changes: " + brief.revision.changes
                if brief.revision
                else brief.brief
            )
            review = critique(
                review_dir, qa, direction, checks=story.get("final_checks", [])
            )
            save(folder / "critique.json", review)
            reasons.extend(qa["blockers"])
            if not qa["passed"] and not reasons:
                reasons.append("Technical score under 75")
            score = 0.7 * qa["score"] + 0.3 * sum(scores) / len(scores)
            if review["status"] == "assessed":
                score = score * 0.75 + review["brief_fit"] * 0.25
                reasons.extend(review["severe_visual_defects"])
                reasons.extend(review.get("failed_checks", []))
                if review["brief_fit"] < 55:
                    reasons.append("Low brief alignment")
            elif story.get("final_checks"):
                reasons.append(
                    "Required final visual checks unavailable: "
                    + review.get("reason", "unknown")
                )
            level = review["status"]
        else:
            score = 0
            level = "no master"
        renderer = "+".join(dict.fromkeys(providers)) or events[-1]["provider"]
        row = {
            "variant": vi + 1,
            "angle": story["angle"],
            "provider": renderer,
            "score": round(score, 1),
            "passed": not reasons,
            "reasons": reasons,
            "vision": level,
            "seconds": round(time.monotonic() - start, 1),
        }
        ranking.append(row)
        query(
            "INSERT OR REPLACE INTO takes VALUES(?,?,?,?,?,?,?)",
            (
                job,
                vi + 1,
                renderer,
                story["angle"],
                row["score"],
                int(row["passed"]),
                row["seconds"],
            ),
        )
        save(root / "ranking.json", ranking)
    rank = sorted(
        ranking,
        key=lambda r: (r["passed"], r["vision"] == "assessed", r["score"]),
        reverse=True,
    )
    winner = rank[0] if rank[0]["passed"] else None
    if winner:
        shutil.copyfile(
            root / f"v{winner['variant']}" / "master.mp4", root / "winner.mp4"
        )
    save(root / "result.json", {"winner": winner, "ranking": rank})
    record(
        job,
        "complete" if winner else "needs_review",
        "Ready" if winner else "No variant passed the quality checks",
    )
