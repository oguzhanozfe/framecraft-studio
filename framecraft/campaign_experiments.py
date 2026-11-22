"""Render and package matched UGC variants from approved footage and measured talking takes."""

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
import wave
from html import escape
from pathlib import Path
from urllib.parse import quote

import numpy as np
from filelock import FileLock

from .core import read, save
from .projects import archive


def matrix(products):
    for product in products:
        for hook in range(product["hooks"]):
            for cta in range(2):
                yield product, hook, cta


def fingerprint(path):
    with path.open("rb") as data:
        return hashlib.file_digest(data, "sha256").hexdigest()


def read_wav(path):
    with wave.open(str(path), "rb") as wav:
        if wav.getparams()[:3] != (1, 2, 24000):
            raise ValueError(f"Expected mono 24 kHz PCM16 speech in {path}")
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")


def assemble_voice(folder, events, target, duration=30):
    rate = 24000
    mix = np.zeros(round(rate * duration), dtype="<i2")
    timeline = []
    end = 0
    for at, key, label in sorted(events):
        samples = read_wav(folder / (key + ".wav"))
        start = round(at * rate)
        stop = start + len(samples)
        if start < end or stop > len(mix) or not len(samples):
            raise ValueError(f"Narration does not fit at {at:g}s: {key}")
        mix[start:stop] = samples
        end = stop
        timeline.append({"start": at, "end": stop / rate, "text": label, "asset": key})
    with wave.open(str(target), "wb") as dst:
        dst.setparams((1, 2, rate, 0, "NONE", ""))
        dst.writeframes(mix.tobytes())
    return timeline


def verified_master(recipe_path):
    spec = read(recipe_path)
    report = read(recipe_path.parent / "motion-assessment.json")
    media = recipe_path.parent / spec["output"]
    if (
        not report["passed"]
        or report["frames"] != 900
        or fingerprint(media) != report["sha256"]
    ):
        raise ValueError(f"The base picture needs a current 30-second audit: {media}")
    return spec, report


def talking_take(folder, key):
    path = folder / (key + "-face.mp4")
    with FileLock(str(path) + ".lock"):
        check = read(folder / (key + "-sync.json"))
        signature = fingerprint(path)
        if (
            not check.get("passed")
            or check.get("video_sha256") != signature
            or check.get("audio_sha256") != fingerprint(folder / (key + ".wav"))
        ):
            raise ValueError(
                f"The selected speaking take has no matching lip-sync check: {key}"
            )
        # Other editions can select a new reading while FFmpeg still opens shots.
        # Bind the timeline to immutable bytes until the whole encode finishes.
        versions = folder / ".takes"
        versions.mkdir(exist_ok=True)
        frozen = versions / (signature + ".mp4")
        if not frozen.is_file():
            with tempfile.TemporaryDirectory(dir=versions) as temp:
                stage = Path(temp) / "take.mp4"
                try:
                    os.link(path, stage)
                except OSError:
                    shutil.copyfile(path, stage)
                os.replace(stage, frozen)
        if fingerprint(frozen) != signature:
            raise ValueError(f"Staged take differs from its signed selection: {key}")
    return frozen, check


def selected_picture(folder, sha256):
    revision = folder / "required-picture.json"
    if not revision.is_file():
        return True
    approved = folder / read(revision)["report"]
    picture = read(approved) if approved.is_file() else {}
    return bool(picture.get("passed") and sha256 and picture.get("sha256") == sha256)


def edition(root, product, hook, cta):
    from .campaign_motion import render

    name = product["slug"]
    folder = root / name
    speech = folder / "speech"
    out = folder / f"H{hook + 1}-C{cta + 1}"
    out.mkdir(parents=True, exist_ok=True)
    with FileLock(str(out / ".lock")):
        base = (root / product["base"]).resolve()
        spec, original = verified_master(base)
        if not selected_picture(folder, original["sha256"]):
            raise ValueError(
                f"Source picture is awaiting approval or has been superseded: {base}"
            )
        lines = read(folder / "final-copy.json")
        key1, key2 = f"hook{hook}", f"cta{cta}"
        opening, first = talking_take(speech, key1)
        ending, last = talking_take(speech, key2)
        for role, receipt in (("opening", first), ("closing", last)):
            if product.get("presenter") and receipt.get("source_sha256"):
                source = product["presenter"][role]
                if (
                    fingerprint(root / source["file"]) != receipt["source_sha256"]
                    or abs(source.get("start", 0) - receipt["source_start"]) > 0.001
                    or abs(source["seconds"] - receipt["seconds"]) > 0.001
                ):
                    raise ValueError(f"Presenter footage changed: {name}/{role}")
        shots = []
        for shot, cue in zip(spec["shots"], original["timeline"], strict=True):
            shot = dict(shot)
            shot["file"] = str((base.parent / shot["file"]).resolve())
            shot["start"] = cue["source_start_seconds"]
            shot.pop("search_radius", None)
            if shot.get("inset"):
                shot["inset"] = dict(shot["inset"])
                shot["inset"]["file"] = str(
                    (base.parent / shot["inset"]["file"]).resolve()
                )
            shots.append(shot)
        for idx, path, label in [
            (0, opening, lines["hooks"][hook]),
            (-1, ending, lines["ctas"][cta]),
        ]:
            seconds = shots[idx]["seconds"]
            shots[idx] = {
                "file": str(path),
                "start": 0,
                "seconds": seconds,
                "layout": "caption",
                "headline": label,
                "check_camera": True,
            }
        last_start = round(30 - shots[-1]["seconds"], 6)
        events = [
            (first.get("voice_onset", 0.25), key1, lines["hooks"][hook]),
            (last_start + last.get("voice_onset", 0.25), key2, lines["ctas"][cta]),
        ]
        for k, at in product["middle_speech"].items():
            events.append((at, k, lines[k] if k in lines else product["brand"]))
        speech_receipt = speech / "creative-ready.json"
        if speech_receipt.is_file():
            readings = read(speech_receipt)["readings"]
            for _, key, line in events:
                if readings[key]["text"] != line or readings[key][
                    "sha256"
                ] != fingerprint(speech / (key + ".wav")):
                    raise ValueError(f"Narration changed since casting: {name}/{key}")
        score = (root / product["score"]).resolve()
        sig = hashlib.sha256(
            json.dumps(
                [
                    original["sha256"],
                    shots,
                    events,
                    [fingerprint(speech / (event[1] + ".wav")) for event in events],
                    first["video_sha256"],
                    last["video_sha256"],
                    fingerprint(score),
                ],
                sort_keys=True,
                ensure_ascii=False,
            ).encode()
        ).hexdigest()
        delivery = out / "delivery.json"
        if delivery.is_file():
            old = read(delivery)
            if (
                old.get("signature") == sig
                and Path(old["archive"]).is_file()
                and fingerprint(Path(old["archive"])) == old["sha256"]
            ):
                return old
        cues = assemble_voice(speech, events, out / "voice.wav")
        recipe = {
            "output": "master.mp4",
            "silent": False,
            "audio_mode": "voice_music",
            "voice": "voice.wav",
            "score": str(score),
            "voice_provider": "Pocket TTS Turkish",
            "voice_profile": product["voice"],
            "spoken": " ".join(c["text"] for c in cues),
            "voice_cues": cues,
            "verify_voice_alignment": True,
            "asr_model": str((root / product["asr"]).resolve()),
            "shots": shots,
            "evidence": spec.get("evidence", []),
        }
        save(out / "motion.json", recipe)
        video = render(recipe, out)
        if not selected_picture(folder, original["sha256"]):
            raise ValueError(f"Source picture changed during rendering: {base}")
        quality = read(out / "motion-assessment.json")
        if (
            not quality["passed"]
            or quality["frames"] != 900
            or not quality["audio"]["voice_alignment"]["passed"]
        ):
            raise ValueError("Failed final picture or audio verification")
        quality["lip_sync"] = {"opening": first, "closing": last}
        quality["experiment"] = {
            "product": name,
            "hook": hook + 1,
            "cta": cta + 1,
            "pair": f"{name}-H{hook + 1}",
            "test_variable": "spoken and written closing call to action",
            "source_picture_sha256": original["sha256"],
        }
        save(out / "motion-assessment.json", quality)
        master = archive(
            spec.get("project", product["project"]),
            video,
            report=out / "motion-assessment.json",
            ledger=out / "motion-frames.jsonl.gz",
        )
        record = {
            "slug": name,
            "project": product["project"],
            "audience": lines.get("audience", ""),
            "hook": hook + 1,
            "cta": cta + 1,
            "hook_text": lines["hooks"][hook],
            "cta_text": lines["ctas"][cta],
            "archive": str(master),
            "sha256": quality["sha256"],
            "signature": sig,
        }
        save(delivery, record)
        return record


def verify_edition(folder, detector, model_dir):
    from .campaign_sync import measure

    rec = read(folder / "delivery.json")
    video = Path(rec["archive"])
    assessment = read(folder / "motion-assessment.json")
    recipe = read(folder / "motion.json")
    digest = fingerprint(video)
    if digest != rec["sha256"] or assessment["sha256"] != digest:
        raise ValueError(f"The rendered edition changed: {video}")
    from .campaign_cast import alternative_lines
    from .campaign_face_integrity import assess as face_integrity
    from .campaign_speech_controls import identify
    from .campaign_speech_review import review

    shots = recipe["shots"]
    cue_by_key = {cue["asset"]: cue for cue in recipe["voice_cues"]}
    speech = folder.parent / "speech"
    readings = speech / "creative-ready.json"
    readings = read(readings)["readings"] if readings.is_file() else {}
    catalog = read(folder.parent.parent / "experiments.json")
    product = next(p for p in catalog["products"] if p["slug"] == rec["slug"])
    timing, articulation, identity, integrity = {}, {}, {}, {}
    for role, key, start, seconds in [
        ("opening", f"hook{rec['hook'] - 1}", 0.0, shots[0]["seconds"]),
        (
            "closing",
            f"cta{rec['cta'] - 1}",
            30.0 - shots[-1]["seconds"],
            shots[-1]["seconds"],
        ),
    ]:
        presenter = product["presenter"][role]
        source = (folder.parent.parent / presenter["file"]).resolve()
        if fingerprint(source) != assessment["lip_sync"][role]["source_sha256"]:
            raise ValueError(f"Reference face changed since rendering: {source}")
        integrity[role] = face_integrity(
            source,
            video,
            start=presenter.get("start", 0.0),
            video_start=start,
            seconds=seconds,
        )
        voice = folder / recipe["voice"]
        timing[role] = measure(video, voice, detector, model_dir, start, seconds, start)
        cue = cue_by_key[key]
        transcript = readings.get(key, {}).get("spoken", cue["text"])
        native = speech / (key + "-sync.json")
        record = read(native) if native.is_file() else {}
        anchors = None
        native_anchors = speech / (key + "-anchors.json")
        if native_anchors.is_file() and record.get("voice_onset") is not None:
            saved = read(native_anchors)
            bed = speech / (key + "-fixed-bed.wav")
            # Reuse stamps only when both the actual source recording and its
            # position in the rendered dialogue timeline match the measured bed.
            if (
                bed.is_file()
                and fingerprint(bed)
                == saved.get("bed_sha256")
                == record.get("bed_sha256")
                and fingerprint(speech / (key + ".wav")) == record.get("audio_sha256")
                and abs(start + record["voice_onset"] - cue["start"]) < 0.001
            ):
                anchors = saved["anchors"]
        articulation[role] = review(
            video,
            voice,
            start=start,
            seconds=seconds,
            audio_start=start,
            transcript=transcript,
            anchors=anchors,
        )
        foils = alternative_lines(speech, key)
        identity[role] = identify(
            video,
            voice,
            foils,
            detector,
            model_dir,
            start=start,
            seconds=seconds,
            audio_start=start,
        )
    check = {
        "version": 2,
        "passed": all(
            result["passed"]
            for group in (timing, articulation, identity, integrity)
            for result in group.values()
        )
        and assessment["audio"]["voice_alignment"]["passed"],
        "sha256": digest,
        "measurement": "Final decoded picture, actual dialogue timeline and separately verified AAC mix. Includes vowel and consonant shapes, silent movement and a different utterance by the same presenter.",
        "assessment": timing,
        "articulation": articulation,
        "spoken_identity": identity,
        "face_integrity": integrity,
    }
    save(folder / "final-lips.json", check)
    save(video.parent / "lip-sync.json", check)
    return check


def verified_print(work, video_sha256):
    requirements = work.parent / "required-print.json"
    if not requirements.is_file():
        return True
    for cue in read(requirements):
        report = work / cue["report"]
        if not report.is_file():
            return False
        audit = read(report)
        frames = cue["frames"]
        if not (
            audit.get("passed")
            and audit.get("video_sha256") == video_sha256
            and audit.get("source_sha256") == cue["reference_sha256"]
            and frames > 0
            and audit.get("frames") == frames
            and audit.get("start_frame") == cue["start_frame"]
            and audit.get("matches", {}).get(cue["phrase"], 0) / frames
            >= cue["minimum_fraction"]
        ):
            return False
    return True


def current_takes(work, hook, cta):
    report = work / "motion-assessment.json"
    rendered = read(report).get("lip_sync", {}) if report.is_file() else {}
    for role, key in (("opening", f"hook{hook}"), ("closing", f"cta{cta}")):
        decision = work.parent / "speech" / (key + "-sync.json")
        selected = read(decision) if decision.is_file() else {}
        if (
            not selected.get("passed")
            or not selected.get("video_sha256")
            or selected["video_sha256"] != rendered.get(role, {}).get("video_sha256")
        ):
            return False
    return True


def export(root, catalog, destination, *, ready_only=False):
    from .campaign_face_integrity import FACE_INTEGRITY_VERSION

    entries = []
    pairs = {}
    for product, h, c in matrix(catalog["products"]):
        work = root / product["slug"] / f"H{h + 1}-C{c + 1}"
        if ready_only and not (work / "delivery.json").is_file():
            continue
        rec = read(work / "delivery.json")
        if product.get("presenter") and not current_takes(work, h, c):
            if ready_only:
                continue
            raise ValueError(f"Superseded or missing presenter selection: {work}")
        if not verified_print(work, rec["sha256"]):
            if ready_only:
                continue
            raise ValueError(f"Final product label verification missing: {work}")
        revision = work.parent / "required-picture.json"
        if revision.is_file():
            audit = work / "motion-assessment.json"
            used = (
                read(audit).get("experiment", {}).get("source_picture_sha256")
                if audit.is_file()
                else None
            )
            if not selected_picture(work.parent, used):
                if ready_only:
                    continue
                raise ValueError(f"Pending revised product picture: {work}")
        segment_file = work.parent / "segments.json"
        if segment_file.is_file():
            segment = read(segment_file)[h]
            if segment["hook_text"] != rec["hook_text"]:
                raise ValueError(f"Audience plan refers to a different opening: {work}")
            rec.update(audience=segment["audience"], hypothesis=segment["hypothesis"])
        src = Path(rec["archive"])
        if (rec["slug"], rec["hook"], rec["cta"]) != (
            product["slug"],
            h + 1,
            c + 1,
        ) or fingerprint(src) != rec["sha256"]:
            raise ValueError(f"Unverified edition: {src}")
        if catalog.get("require_final_lips", True):
            if ready_only and not (work / "final-lips.json").is_file():
                continue
            sync = read(work / "final-lips.json")
            valid = (
                sync.get("version") == 2
                and sync.get("passed")
                and sync.get("sha256") == rec["sha256"]
                and all(
                    sync.get("face_integrity", {}).get(role, {}).get("version")
                    == FACE_INTEGRITY_VERSION
                    for role in ("opening", "closing")
                )
                and all(
                    sync.get(group, {}).get(role, {}).get("passed") is True
                    for group in ("articulation", "spoken_identity", "face_integrity")
                    for role in ("opening", "closing")
                )
            )
            if not valid:
                if ready_only:
                    continue
                raise ValueError(f"Final audiovisual verification missing: {src}")
        entries.append(rec)
        pairs.setdefault((rec["slug"], h + 1), []).append(rec)
    if ready_only:
        pairs = {key: pair for key, pair in pairs.items() if len(pair) == 2}
        entries = [entry for pair in pairs.values() for entry in pair]
    if len({x["sha256"] for x in entries}) != len(entries):
        raise ValueError("Duplicate deliverables in experiment")
    destination.mkdir(parents=True, exist_ok=True)
    old_manifest = destination / "manifest.json"
    previous = read(old_manifest) if old_manifest.is_file() else []
    cards = []
    with tempfile.TemporaryDirectory(dir=destination) as cur:
        stage = Path(cur)
        for (slug, h), pair in pairs.items():
            if (
                len(pair) != 2
                or pair[0]["hook_text"] != pair[1]["hook_text"]
                or pair[0]["cta_text"] == pair[1]["cta_text"]
            ):
                raise ValueError(f"Invalid control pair: {slug} {h}")
            content = []
            preview = ""
            first_dir = root / slug / f"H{h}-C1"
            if (first_dir / "motion-contact.jpg").is_file():
                import cv2

                preview_dir = destination / "previews"
                preview_dir.mkdir(exist_ok=True)
                thumbnail = preview_dir / (pair[0]["sha256"][:20] + ".jpg")
                if not thumbnail.exists():
                    camera = cv2.VideoCapture(pair[0]["archive"])
                    try:
                        camera.set(cv2.CAP_PROP_POS_MSEC, 1000)
                        ok, frame = camera.read()
                    finally:
                        camera.release()
                    if ok:
                        thumb = cv2.resize(
                            frame, (288, 512), interpolation=cv2.INTER_AREA
                        )
                        tmp = stage / thumbnail.name
                        if cv2.imwrite(str(tmp), thumb):
                            os.replace(tmp, thumbnail)
                if thumbnail.is_file():
                    preview = f' poster="previews/{thumbnail.name}"'
            for rec in pair:
                path = Path(rec["archive"])
                name = f"{slug}-H{h}-{'AB'[rec['cta'] - 1]}.mp4"
                dst = destination / name
                if not dst.exists() or fingerprint(dst) != rec["sha256"]:
                    tmp = stage / name
                    try:
                        os.link(path, tmp)
                    except OSError:
                        shutil.copy2(path, tmp)
                    os.replace(tmp, dst)
                rec["file"] = name
                content.append(
                    f'<article><video controls preload="none" playsinline{preview} src="{quote(name)}"></video><p><b>{"AB"[rec["cta"] - 1]}</b> · {escape(rec["cta_text"])}</p></article>'
                )
            cards.append(
                f"<section><h2>{escape(pair[0]['project'])} · Açılış {h}</h2><p>“{escape(pair[0]['hook_text'])}”</p><p>Hedef kitle · {escape(pair[0]['audience'])}</p><div>{''.join(content)}</div></section>"
            )
        with (stage / "ab-testleri.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as file:
            keys = [
                "slug",
                "project",
                "hook",
                "cta",
                "hook_text",
                "cta_text",
                "audience",
                "hypothesis",
                "file",
                "sha256",
            ]
            writer = csv.DictWriter(file, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(entries)
        save(stage / "manifest.json", entries)
        (stage / "index.html").write_text(
            '<!doctype html><html lang="tr"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>UGC Creator · A/B</title><style>body{font:16px system-ui;background:#141719;color:#eee;max-width:950px;margin:auto;padding:1rem}section{padding:1rem 0;border-top:1px solid #444}section div{display:grid;grid-template-columns:1fr 1fr;gap:1rem}video{width:100%;max-height:54vh;background:#101010}a{color:#bce8f8}</style><h1>UGC Creator · A/B</h1><p>Her satırda aynı açılış ve iki farklı kapanış var. Çiftleri aynı hedef kitle, yerleşim ve bütçeyle karşılaştırın.</p><a href="ab-testleri.csv">Test tablosu</a>'
            + "".join(cards)
            + "</html>",
            encoding="utf8",
        )
        for name in ("ab-testleri.csv", "manifest.json", "index.html"):
            os.replace(stage / name, destination / name)
    listed = {row["file"] for row in entries}
    for old in previous:
        name = old["file"]
        if name in listed or Path(name).name != name or not name.endswith(".mp4"):
            continue
        copy, archive_path = destination / name, Path(old["archive"])
        # Withdraw only a link created from the immutable archive; keep unique local files.
        if (
            copy.is_file()
            and archive_path.is_file()
            and not archive_path.resolve().is_relative_to(destination.resolve())
            and os.path.samefile(copy, archive_path)
        ):
            copy.unlink()
    return len(entries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", type=Path)
    parser.add_argument("--product")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--export", type=Path)
    mode.add_argument(
        "--verify-lips", nargs=2, metavar=("DETECTOR", "MODEL_DIR"), type=Path
    )
    parser.add_argument(
        "--ready-only", action="store_true", help="Export only complete reviewed pairs"
    )
    args = parser.parse_args()
    cat = read(args.catalog)
    if args.ready_only and not args.export:
        parser.error("--ready-only requires --export")
    if args.export:
        print(export(args.catalog.parent, cat, args.export, ready_only=args.ready_only))
    else:
        for prod, h, c in matrix(cat["products"]):
            if args.product is None or prod["slug"] == args.product:
                if args.verify_lips:
                    folder = args.catalog.parent / prod["slug"] / f"H{h + 1}-C{c + 1}"
                    if not verify_edition(folder, *args.verify_lips)["passed"]:
                        raise SystemExit(f"Lip-sync review failed: {folder}")
                    print(folder / "final-lips.json", flush=True)
                else:
                    print(
                        edition(args.catalog.parent, prod, h, c)["archive"], flush=True
                    )
