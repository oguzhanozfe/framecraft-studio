"""Build campaign shots, voice, evidence and a reviewable master using local models."""

import hashlib
import os
import secrets
import shutil
import cv2
import numpy as np
from filelock import FileLock
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

from .campaign_edit import preview_cfr
from .campaign_motion import render
from .core import read, save
from .director import CREATOR_DIRECTION, CreativeIntent, ask
from .motion_copy import make_voice
from .motion_review import reference_check
from .providers import generate, inventory


class Scene(BaseModel):
    action: str = Field(min_length=30, max_length=500)
    label: str = Field(min_length=2, max_length=32)


class Scenes(BaseModel):
    intent: CreativeIntent
    shots: list[Scene] = Field(min_length=6, max_length=6)


class CleanViews(BaseModel):
    photos: list[int] = Field(min_length=1, max_length=4)
    reason: str


def refs(item, source):
    files = sorted(source.glob("src-*.jpg"))
    if not files:
        return []
    cache = source / "motion-views.json"
    key = [(f.name, f.stat().st_size) for f in files]
    if cache.is_file() and read(cache)["signature"] == [list(v) for v in key]:
        return [source / name for name in read(cache)["use"]]
    tiles = []
    for idx, file in enumerate(files):
        img = cv2.imread(str(file))
        h, w = img.shape[:2]
        scale = min(176 / w, 206 / h)
        thumb = cv2.resize(
            img,
            (max(1, round(w * scale)), max(1, round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
        tile = np.full((240, 188, 3), 45, np.uint8)
        x = (188 - thumb.shape[1]) // 2
        tile[23 : 23 + thumb.shape[0], x : x + thumb.shape[1]] = thumb
        cv2.putText(
            tile,
            str(idx + 1),
            (7, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.57,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    board = np.vstack([np.hstack(tiles[i : i + 3]) for i in range(0, len(tiles), 3)])
    views = ask(
        CleanViews,
        "Select numbered pictures that could be the first instant of real product footage. Prefer in-use photos or clean views where the one physical SKU is dominant. Reject publicity panels, color blocks with explanatory text, collages, diagrams, packaging and feature tables whenever any clean view exists. A tiny label printed on the physical item is fine. Return a single choice if the others have any promotional text. Return each chosen number once.",
        {"product": item["name"], "number_of_choices": len(files)},
        cv2.imencode(".jpg", board)[1].tobytes(),
        model_override=item.get("vision_model"),
    )
    chosen = [files[i - 1] for i in dict.fromkeys(views.photos) if 1 <= i <= len(files)]
    if chosen:
        save(
            cache,
            {"signature": key, "use": [f.name for f in chosen], "reason": views.reason},
        )
        return chosen
    picked = []
    for s in item.get("gallery", []):
        if s.get("photo"):
            path = source / f"src-{s['photo']:02}.jpg"
            if path.is_file() and path not in picked:
                picked.append(path)
    return picked or files


def portrait_anchor(path, out):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        color = im.getpixel((0, 0))
        stage = Image.new("RGB", (576, 1024), color)
        im.thumbnail((576, 1024), Image.Resampling.LANCZOS)
        stage.paste(im, ((576 - im.width) // 2, (1024 - im.height) // 2))
        stage.save(out)


def moving_interval(video, seconds, *, skip=0):
    """Choose an uninterrupted interval after any static reference lead-in."""
    cap = cv2.VideoCapture(str(video))
    active = []
    last = None
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h = frame.shape[0]
            thumb = cv2.resize(
                cv2.cvtColor(frame[h // 9 : h * 2 // 3], cv2.COLOR_BGR2GRAY),
                (24, 28),
                interpolation=cv2.INTER_AREA,
            )
            active.append(last is None or np.mean(cv2.absdiff(thumb, last)) >= 0.18)
            last = thumb
    finally:
        cap.release()
    if fps <= 0:
        raise ValueError(f"Unreadable video: {video}")
    length = round(seconds * fps)
    latest = len(active) - length
    preferred = min(latest, round(skip * fps))
    for start in sorted(range(latest + 1), key=lambda i: abs(i - preferred)):
        still = 0
        for changed in active[start + 1 : start + length]:
            still = 0 if changed else still + 1
            if still >= max(2, round(fps * 0.45)):
                break
        else:
            return start / fps
    return None


def source_shots(item, source, output, scenes):
    shots = []
    provider = item.get("render_model")
    if not provider or provider == "auto":
        provider = next(
            (
                model["id"]
                for model in inventory()["models"]
                if model["available"] and model["id"] in {"h3", "ltx", "wan"}
            ),
            None,
        )
    if not provider:
        raise ValueError("Connect H3, LTX or Wan in ComfyUI for the presenter shots")
    game = item["kind"] != "product"
    publisher = item.get("publisher_footage", {})
    real = (
        (source / publisher["file"] if publisher else preview_cfr(source))
        if game
        else None
    )
    photos = refs(item, source) if not game else []
    durations = [4, 5, 6, 5, 6, 4] if game else [4, 6, 6, 5, 5, 4]
    for n, (s, duration) in enumerate(zip(scenes, durations, strict=True)):
        role = "creative" if n in (0, 5) else "demonstration"
        supplied = item.get("shot_sources", {}).get(str(n + 1)) or (
            item.get("creator_sources", {}).get(str(n + 1))
            if role == "creative"
            else None
        )
        if supplied:
            path = source.parent / supplied
            reference = (
                photos[(n - 1) % len(photos)]
                if photos and role == "demonstration"
                else None
            )
            checked = reference_check(path, reference, model=item.get("vision_model"))
            start = moving_interval(path, duration)
            if checked["defect"] != "none" or start is None:
                supplied = None
            else:
                framing = "cover"
        if not supplied and game and role == "demonstration":
            # Leave the title/end slates of publisher reels out of the cut.
            start = publisher["starts"][n - 1] if publisher else 2 + sum(durations[1:n])
            path = real
            framing = publisher.get("framing") or (
                "cover" if item.get("portrait_game") else "contain"
            )
        elif not supplied:
            frame_folder = output / f"take-{n + 1}"
            frame_folder.mkdir(exist_ok=True)
            path = frame_folder / "motion.mp4"
            start = 0
            framing = "cover"
            ref = None
            if not game and role == "demonstration":
                if not photos:
                    raise ValueError("Missing SKU reference")
                ref = frame_folder / "reference.png"
                portrait_anchor(photos[(n - 1) % len(photos)], ref)
            prompt = s["action"]
            receipt = frame_folder / "take.json"
            sig = hashlib.sha256(
                prompt.encode() + (ref.read_bytes() if ref else b"")
            ).hexdigest()
            if (
                not receipt.exists()
                or read(receipt)["signature"] != sig
                or read(receipt).get("model") != provider
                or not path.exists()
            ):
                seed = secrets.randbelow(2**31 - 1)
                tmp = frame_folder / "take.partial.mp4"
                generate(
                    provider,
                    prompt,
                    seed,
                    "vertical",
                    tmp,
                    print,
                    frames=(duration + (1 if ref else 0)) * 24 + 1,
                    reference=ref,
                )
                tmp.replace(path)
                save(
                    receipt,
                    {
                        "seed": seed,
                        "model": provider,
                        "signature": sig,
                        "prompt": prompt,
                        "reference": ref.name if ref else None,
                    },
                )
            for attempt in range(3):
                checked = reference_check(path, ref, model=item.get("vision_model"))
                start = moving_interval(path, duration, skip=0.8 if ref else 0)
                if start is None:
                    checked = {
                        "defect": "motion_missing",
                        "observation": "The shot freezes before the intended cut. Keep moving continuously through the shot.",
                        "location": "within the selected source interval",
                    }
                if checked["defect"] == "none":
                    break
                if attempt == 2:
                    raise ValueError(
                        f"Take {n + 1} still fails visual review: {checked['defect']} {checked['location']}"
                    )
                revision = (
                    prompt
                    + " Correct the following observed defect in this take: "
                    + checked["observation"]
                    + " "
                    + checked["location"]
                )
                seed = secrets.randbelow(2**31 - 1)
                tmp = frame_folder / "take.partial.mp4"
                generate(
                    provider,
                    revision,
                    seed,
                    "vertical",
                    tmp,
                    print,
                    frames=(duration + (1 if ref else 0)) * 24 + 1,
                    reference=ref,
                )
                tmp.replace(path)
                info = read(receipt)
                info.update(seed=seed, prompt=revision)
                info.setdefault("repairs", []).append(checked)
                save(receipt, info)
        shots.append(
            {
                "file": os.path.relpath(path, output),
                "seconds": duration,
                "start": start,
                "framing": framing,
                **(
                    {
                        "search_radius": publisher.get("search_radius", 3),
                        "avoid_freeze": True,
                    }
                    if game and role == "demonstration"
                    else {}
                ),
                "layout": (
                    "caption"
                    if item.get("silent")
                    else ("corner", "tag", "clean", "side", "clean", "corner")[n]
                ),
                "detail": (item["name"] if n == 0 and item.get("silent") else ""),
                "eyebrow": item["name"] if n in (0, 5) else "",
                "headline": s["label"],
            }
        )
    return shots


def produce(item, parent, *, stage="run"):
    output = parent / "motion"
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / "campaign.lock"), timeout=3600):
        return produce_locked(item, parent, output, stage)


def produce_locked(item, parent, output, stage):
    custom = item.get("motion_recipe")
    if stage == "verify":
        from .campaign_motion import audit

        spec = (parent / custom).resolve() if custom else output / "motion.json"
        recipe = read(spec)
        master = spec.parent / recipe.get("output", "motion.mp4")
        prior = read(spec.parent / "motion-assessment.json")
        result = audit(master, spec.parent, prior["timeline"], recipe)
        if not result["passed"]:
            raise ValueError("Motion QA: " + ", ".join(result["failures"]))
        return master
    if custom:
        config = (parent / custom).resolve()
        if stage == "prepare":
            from .campaign_motion import spans

            spans(read(config), config.parent)
            return config
        with FileLock(str(config.parent / "motion.lock"), timeout=3600):
            return render(read(config), config.parent)
    source = parent / "source"
    evidence = read(source / "source.json")
    brief = {
        "product": item["name"],
        "audience": item["audience"],
        "direction": item["angle"],
        "facts": item["proof"],
        "pronunciation": item.get("pronunciation", {}),
        "writer": item.get("editor_model"),
        "judge": item.get("writer_model"),
    }
    voice_file = output / "voice.json"
    silent = item.get("silent", False)
    if silent:
        receipt = {"copy": item["angle"]}
    else:
        if (
            not voice_file.exists()
            or not (output / "voice.wav").is_file()
            or not read(voice_file).get("segments")
            or read(voice_file)["source"]
            != {k: brief[k] for k in ("product", "audience", "direction", "facts")}
        ):
            make_voice(output, brief)
        receipt = read(voice_file)
    plan = output / "shots.json"
    if (
        not plan.exists()
        or read(plan).get("read") != receipt["copy"]
        or not read(plan).get("intent")
    ):
        result = ask(
            Scenes,
            CREATOR_DIRECTION
            + (
                "Tell the story in pictures and in natural, compact Turkish captions. No one mouths dialogue. First and last shots show an adult creator in an ordinary setting with natural lighting. Use straight cuts, small plausible gestures and stable backgrounds. "
                if silent
                else ""
            )
            + "Direct six chronological portrait shots for a Turkish creator video. Give each one a distinct 2–4 word Turkish on-screen headline relevant to the spoken read. For each action write a concrete English single-take smartphone camera direction. Open and close with an adult creator engaged in an activity directly relevant to the selling point, no dialogue or visible screen text. Middle four shots of a retail item are reference-conditioned: preserve the exact product and mark placement, show a single physically plausible interaction or real change of viewpoint per shot. Garments use a professional adult fashion presentation. For a game the middle four shots use actual publisher gameplay. Avoid lifestyle scenes that could introduce any unrelated product. Avoid simulated handwriting, overlays, zoom-only motion, staged smiles and unverified claims.",
            {
                "item": item["name"],
                "type": item["kind"],
                "facts": item["proof"],
                "script": receipt["copy"],
                "existing_shot_actions": item.get("shot_actions", {}),
                "audience": item["audience"],
            },
            model_override=item.get("editor_model"),
        )
        save(
            plan,
            {"read": receipt["copy"], **result.model_dump()},
        )
    if stage == "prepare":
        return plan
    shots = source_shots(item, source, output, read(plan)["shots"])
    recipe = {
        "shots": shots,
        "silent": silent,
        "project": item.get("project") or item["name"],
        "output": "master.mp4",
        "evidence": [evidence["url"]],
    }
    if item.get("publisher_footage"):
        recipe["evidence"].append(item["publisher_footage"]["url"])
    if not silent:
        recipe.update(
            voice="voice.wav",
            spoken=receipt["text"],
            voice_provider=receipt["provider"],
        )
    if not silent and item.get("asr_model"):
        recipe["asr_model"] = os.path.relpath(parent.parent / item["asr_model"], output)
    if not silent and item.get("score_direction"):
        from .campaign_score import compose

        score = output / "score.flac"
        receipt_path = score.with_suffix(".json")
        if (
            not score.is_file()
            or not receipt_path.is_file()
            or read(receipt_path).get("prompt") != item["score_direction"]
        ):
            compose(
                item["score_direction"],
                score,
                seed=int(hashlib.sha256(item["slug"].encode()).hexdigest()[:7], 16),
            )
        recipe["score"] = score.name
    save(output / "motion.json", recipe)
    return render(recipe, output)


def export_motion(root, items):
    import html
    import zipfile

    delivery = root / "motion-delivery"
    delivery.mkdir(exist_ok=True)
    cards = []
    catalog = []
    files = []
    for item in items:
        folder = root / item["slug"]
        recipe = (
            (folder / item["motion_recipe"]).resolve()
            if item.get("motion_recipe")
            else folder / "motion" / "motion.json"
        )
        spec = read(recipe)
        report = read(recipe.parent / "motion-assessment.json")
        target = recipe.parent / spec.get("output", "motion.mp4")
        with target.open("rb") as stream:
            current = hashlib.file_digest(stream, "sha256").hexdigest()
        if not report["passed"] or report["sha256"] != current:
            raise ValueError(f"Missing current reviewed master: {item['slug']}")
        dst = delivery / (item["slug"] + ".mp4")
        shutil.copy2(target, dst)
        files.append(dst)
        catalog.append(
            {
                "product": item["name"],
                "file": dst.name,
                "audience": item["audience"],
                "evidence": spec["evidence"],
                "assessment": report,
            }
        )
        cards.append(
            f'<article><h2>{html.escape(item["name"])}</h2><video controls playsinline preload="metadata" src="{html.escape(dst.name, quote=True)}"></video></article>'
        )
    gallery = delivery / "index.html"
    gallery.write_text(
        '<!doctype html><html lang="tr"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Framecraft UGC</title><style>body{background:#11161a;color:white;font:16px system-ui;margin:4vw}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:24px}video{width:100%;max-height:72vh}h2{font-size:1rem}</style><h1>Framecraft UGC</h1><main>'
        + "".join(cards)
        + "</main></html>",
        encoding="utf-8",
    )
    audit = delivery / "rapor.json"
    save(audit, catalog)
    bundle = root / "Framecraft-UGC-Motion.zip"
    tmp = bundle.with_suffix(".partial")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as z:
        for f in [*files, gallery, audit]:
            z.write(f, f.name)
    tmp.replace(bundle)
