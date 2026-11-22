"""Build and verify a presenter's talking takes from their own-camera footage and voice."""

import argparse
import gc
import hashlib
import json
import os
import shutil
import tempfile
import wave
from pathlib import Path

from filelock import FileLock

from .core import read, save


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def voice_bed(source, target, seconds, onset):
    with wave.open(str(source), "rb") as src:
        if src.getparams()[:3] != (1, 2, 24000):
            raise ValueError(f"Expected a 24 kHz mono PCM16 voice: {source}")
        samples = src.readframes(src.getnframes())
    size = 2 * round(24000 * seconds)
    start = 2 * round(24000 * onset)
    if start < 0 or not samples or start + len(samples) > size:
        raise ValueError(f"The voice at {onset:g}s does not fit its {seconds:g}s shot")
    data = bytearray(size)
    data[start : start + len(samples)] = samples
    with wave.open(str(target), "wb") as dst:
        dst.setparams((1, 2, 24000, 0, "NONE", ""))
        dst.writeframes(data)


def alternative_lines(speech, key):
    """Use other dialogue by this project's cast, close in length to the target."""
    named = [
        p
        for p in speech.glob("*.wav")
        if p.stem != key
        and (
            p.stem.startswith(("hook", "cta"))
            and p.stem[-1:].isdigit()
            or p.stem in ("feature", "use")
        )
    ]
    with wave.open(str(speech / (key + ".wav")), "rb") as f:
        duration = f.getnframes() / f.getframerate()

    def distance(path):
        with wave.open(str(path), "rb") as f:
            return abs(f.getnframes() / f.getframerate() - duration)

    return sorted(named, key=lambda path: (distance(path), path.name))[:2]


def adopt_candidate(source, speech, key, candidate, report, bed, detector, syncnet):
    """Promote only the measured performance for this exact actor and spoken line."""
    from .campaign_face_integrity import assess as face_integrity
    from .campaign_speech_controls import identify
    from .campaign_sync import measure

    receipt = read(candidate.with_suffix(".json"))
    asset = speech / (key + ".wav")
    sig = digest(candidate)
    bed_sig = digest(bed)
    physical = read(report)
    if not (
        physical.get("passed")
        and physical.get("sha256") == sig
        and physical.get("bed_sha256") == bed_sig
        and receipt.get("video_sha256") == sig
        and receipt.get("source_sha256") == digest(source)
        and receipt.get("audio_sha256") == digest(asset)
    ):
        raise ValueError("Candidate, actor, speech or articulation receipt changed")
    seconds = receipt["seconds"]
    # Bind the actual onset and all samples to the original line, including pauses.
    with tempfile.TemporaryDirectory(dir=speech) as temp:
        reference = Path(temp) / "timed.wav"
        voice_bed(asset, reference, seconds, receipt["voice_onset"])
        with (
            wave.open(str(reference), "rb") as expected,
            wave.open(str(bed), "rb") as actual,
        ):
            if expected.getparams()[:4] != actual.getparams()[
                :4
            ] or expected.readframes(expected.getnframes()) != actual.readframes(
                actual.getnframes()
            ):
                raise ValueError(
                    "Timed dialogue differs from the recorded line or onset"
                )
    timing = measure(candidate, bed, detector, syncnet, seconds=seconds)
    foils = alternative_lines(speech, key)
    which = identify(candidate, bed, foils, detector, syncnet, seconds=seconds)
    integrity = face_integrity(
        source, candidate, start=receipt["source_start"], seconds=seconds
    )
    checks = {
        **receipt,
        **timing,
        "passed": timing["passed"] and which["passed"] and integrity["passed"],
        "face_integrity": integrity,
        "articulation": physical,
        "spoken_identity": which,
        "bed_sha256": bed_sig,
        "alternatives": [p.name for p in foils],
        "candidate": receipt.get("provider", candidate.parent.name),
    }
    save(candidate.with_suffix(".selection.json"), checks)
    if not checks["passed"]:
        return checks
    output = speech / (key + "-face.mp4")
    with (
        FileLock(str(output) + ".lock"),
        tempfile.TemporaryDirectory(dir=speech) as tmp,
    ):
        staged = Path(tmp) / "face.mp4"
        shutil.copyfile(candidate, staged)
        if digest(staged) != sig:
            raise ValueError("Speaking take changed during promotion")
        os.replace(staged, output)
        save(speech / (key + "-sync.json"), checks)
    return checks


def checked_picture(
    source,
    video,
    bed,
    detector,
    syncnet,
    start,
    seconds,
    transcript,
    timing,
    *,
    donor=None,
    receipt=None,
):
    """Repair the measured defect, and qualify the exact replacement against the same audio."""
    from .campaign_face_integrity import assess
    from .campaign_speech_review import review
    from .campaign_sync import measure

    shape = review(video, bed, seconds=seconds, transcript=transcript)
    if not shape["passed"] and shape.get("acoustic_tokens"):
        from .campaign_viseme_firm import repair_neural
        from .campaign_visemes import refine

        posed = video.with_name("phonetic-poses.mp4")
        localized = video.with_name("localized-phonemes.mp4")
        refine(video, bed, shape["acoustic_tokens"], posed)
        repair_neural(video, posed, shape["acoustic_tokens"], localized)
        for proposed in (localized, posed):
            measured = review(proposed, bed, seconds=seconds, transcript=transcript)
            if measured["passed"]:
                score = measure(proposed, bed, detector, syncnet, seconds=seconds)
                if score["passed"]:
                    video, shape, timing = proposed, measured, score
                    break
    if not shape["passed"]:
        raise ValueError(f"Unverified mouth shapes: {shape['failures']}")
    skin = assess(source, video, start=start, seconds=seconds)
    if not skin["passed"]:
        from .campaign_lip_composite import render as restore

        if (
            donor
            and receipt
            and donor.is_file()
            and donor.with_suffix(".json").is_file()
            and shape.get("acoustic_tokens")
        ):
            from .campaign_viseme_firm import repair_from_report

            localized = video.with_name("localized-lip-texture.mp4")
            bound = {**receipt, "video_sha256": digest(video)}
            try:
                repair_from_report(
                    video,
                    donor,
                    shape["acoustic_tokens"],
                    skin,
                    localized,
                    video_receipt=bound,
                    donor_receipt=read(donor.with_suffix(".json")),
                )
            except ValueError:
                # An earlier audition can have a different actor, onset or duration.
                pass
            else:
                candidate_skin = assess(source, localized, start=start, seconds=seconds)
                candidate_shape = review(
                    localized, bed, seconds=seconds, transcript=transcript
                )
                candidate_timing = measure(
                    localized, bed, detector, syncnet, seconds=seconds
                )
                if all(
                    c["passed"]
                    for c in (candidate_skin, candidate_shape, candidate_timing)
                ):
                    return localized, candidate_shape, candidate_timing, candidate_skin
        corrected = video.with_name("original-lip-texture.mp4")
        restore(source, video, corrected, source_start=start, seconds=seconds)
        skin = assess(source, corrected, start=start, seconds=seconds)
        if not skin["passed"]:
            raise ValueError(
                f"Unresolved facial texture or contour errors: {skin['failure_times']}"
            )
        shape = review(corrected, bed, seconds=seconds, transcript=transcript)
        timing = measure(corrected, bed, detector, syncnet, seconds=seconds)
        if not (shape["passed"] and timing["passed"]):
            raise ValueError(
                "Source-texture repair changed the recorded articulation or timing"
            )
        video = corrected
    return video, shape, timing, skin


def prepare_take(
    network,
    source,
    audio,
    output,
    detector,
    syncnet,
    *,
    start,
    seconds,
    batch=4,
    transcript=None,
    audition=None,
):
    from .campaign_sync import measure

    report = output.with_name(output.stem.removesuffix("-face") + "-sync.json")
    identity = {
        "source_sha256": digest(source),
        "audio_sha256": digest(audio),
        "source_start": start,
        "seconds": seconds,
        "visual_review_version": 2,
        "transcript": transcript,
    }
    from .campaign_face_integrity import FACE_INTEGRITY_VERSION

    if getattr(network, "cache_version", None):
        identity["pipeline"] = network.cache_version
    with FileLock(str(output) + ".lock"):
        if output.is_file() and report.is_file():
            previous = read(report)
            if (
                previous.get("passed")
                and previous.get("face_integrity", {}).get("version")
                == FACE_INTEGRITY_VERSION
                and all(previous.get(k) == v for k, v in identity.items())
                and previous.get("video_sha256") == digest(output)
            ):
                return previous
        with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
            folder = Path(tmp)
            bed, video = folder / "voice.wav", folder / "talking.mp4"
            delay = 0
            for _ in range(3):
                onset = 0.25
                voice_bed(audio, bed, seconds, onset)
                network.run(
                    source,
                    bed,
                    video,
                    start,
                    0,
                    seconds,
                    batch=batch,
                    voice_delay_ms=delay,
                    speech_only=True,
                )
                check = measure(video, bed, detector, syncnet, seconds=seconds)
                offset = check.get("offset_ms", 0)
                if 40 <= abs(offset) <= 200 and check.get("confidence", 0) >= 0.55:
                    aligned = round(onset - offset / 1000, 3)
                    try:
                        voice_bed(audio, bed, seconds, aligned)
                    except ValueError:
                        pass
                    else:
                        trial = measure(video, bed, detector, syncnet, seconds=seconds)
                        if trial["passed"] and abs(trial["offset_ms"]) < abs(offset):
                            onset, check = (
                                aligned,
                                {**trial, "initial_offset_ms": offset},
                            )
                if audition is not None:
                    audition.parent.mkdir(exist_ok=True)
                    with tempfile.TemporaryDirectory(dir=audition.parent) as stage:
                        staged = Path(stage) / "base.mp4"
                        shutil.copyfile(video, staged)
                        os.replace(staged, audition)
                    save(
                        audition.with_suffix(".json"),
                        {
                            **identity,
                            "voice_onset": onset,
                            "video_sha256": digest(audition),
                        },
                    )
                if check["passed"]:
                    from .campaign_speech_controls import identify

                    video, articulation, check, integrity = checked_picture(
                        source,
                        video,
                        bed,
                        detector,
                        syncnet,
                        start,
                        seconds,
                        transcript,
                        check,
                    )
                    alternates = alternative_lines(audio.parent, audio.stem)
                    if alternates:
                        identity_check = identify(
                            video, bed, alternates, detector, syncnet, seconds=seconds
                        )
                        if not identity_check["passed"]:
                            raise ValueError(
                                f"Could not identify the spoken line {audio.stem}: {identity_check['failures']}"
                            )
                    else:
                        identity_check = {
                            "status": "pending",
                            "reason": "no other reading by this presenter",
                        }
                    record = {
                        **identity,
                        **check,
                        "articulation": articulation,
                        "face_integrity": integrity,
                        "spoken_identity": identity_check,
                        "voice_onset": onset,
                        "video_sha256": digest(video),
                    }
                    os.replace(video, output)
                    save(report, record)
                    return record
                if check.get("confidence", 0) < 0.55 or not offset:
                    break
                delay += offset
            raise ValueError(
                f"Rejected speaking take {audio.stem}: {json.dumps(check)}"
            )


def diffusion_fallback(
    source, speech, key, shot, config, detector, syncnet, transcript, *, donor=None
):
    """Generate from the original recording and recheck the actual candidate before use."""
    from .campaign_face_integrity import FACE_INTEGRITY_VERSION
    from .campaign_latentsync import render_registered_take
    from .campaign_sync import measure

    source_hash = digest(source)
    audio = speech / (key + ".wav")
    cached_video = speech / (key + "-face.mp4")
    cached_report = speech / (key + "-sync.json")
    if cached_video.is_file() and cached_report.is_file():
        prior = read(cached_report)
        if (
            prior.get("fallback_config") == config
            and prior.get("transcript") == transcript
            and prior.get("visual_review_version") == 2
            and prior.get("passed")
            and prior.get("face_integrity", {}).get("passed")
            and prior["face_integrity"].get("version") == FACE_INTEGRITY_VERSION
            and prior.get("spoken_identity", {}).get("passed")
            and prior.get("articulation", {}).get("passed")
            and prior.get("source_start") == shot.get("start", 0.0)
            and prior.get("seconds") == shot["seconds"]
            and prior.get("voice_onset") == shot.get("voice_onset", 0.25)
            and prior.get("source_sha256") == source_hash
            and prior.get("audio_sha256") == digest(audio)
            and prior.get("video_sha256") == digest(cached_video)
        ):
            return prior
    with tempfile.TemporaryDirectory(dir=speech) as temp:
        scratch = Path(temp)
        candidate, bed = scratch / "take.mp4", scratch / "timed.wav"
        onset = shot.get("voice_onset", 0.25)
        seconds = shot["seconds"]
        voice_bed(audio, bed, seconds, onset)
        render_registered_take(
            source,
            audio,
            candidate,
            source_start=shot.get("start", 0.0),
            seconds=seconds,
            voice_onset=onset,
            **config,
        )
        generated = read(candidate.with_suffix(".json"))
        timing = measure(candidate, bed, detector, syncnet, seconds=seconds)
        candidate, shape, timing, _ = checked_picture(
            source,
            candidate,
            bed,
            detector,
            syncnet,
            shot.get("start", 0),
            seconds,
            transcript,
            timing,
            donor=donor,
            receipt=generated,
        )
        if not timing["passed"]:
            raise ValueError(f"Unverified diffusion timing for {key}")
        generated.update(
            provider="latent",
            visual_review_version=2,
            fallback_config=config,
            transcript=transcript,
            video_sha256=digest(candidate),
        )
        save(candidate.with_suffix(".json"), generated)
        shape.update(sha256=digest(candidate), bed_sha256=digest(bed))
        evidence = scratch / "articulation.json"
        save(evidence, shape)
        result = adopt_candidate(
            source, speech, key, candidate, evidence, bed, detector, syncnet
        )
        if not result["passed"]:
            raise ValueError(
                f"Diffusion take failed final face or speech checks for {key}"
            )
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument("musetalk", type=Path)
    parser.add_argument("syncnet", type=Path)
    parser.add_argument("--product")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--diffusion-config",
        type=Path,
        help="JSON paths for the optional local LatentSync fallback",
    )
    args = parser.parse_args()
    from .campaign_dub import Dubber

    network = None
    fallback = None
    home = Path(os.environ.get("FRAMECRAFT_HOME", Path.home() / ".framecraft"))
    configuration = args.diffusion_config or home / "models" / "latent-sync.json"
    if args.diffusion_config or configuration.is_file():
        configuration = configuration.resolve()
        supplied = read(configuration)
        fallback = {
            k: (
                str((configuration.parent / v).resolve())
                if k
                in (
                    "repo",
                    "checkpoints",
                    "vae",
                    "model_python",
                    "work_dir",
                    "gpu_lock",
                )
                else v
            )
            for k, v in supplied.items()
        }
    root = args.catalog.resolve().parent
    for product in read(args.catalog)["products"]:
        if args.product and product["slug"] != args.product:
            continue
        speech = root / product["slug"] / "speech"
        casting = speech / "creative-ready.json"
        readings = read(casting).get("readings", {}) if casting.is_file() else {}
        for role, prefix, count in (
            ("opening", "hook", product["hooks"]),
            ("closing", "cta", 2),
        ):
            shot = product["presenter"][role]
            for i in range(count):
                key = f"{prefix}{i}"
                source = root / shot["file"]
                detector = args.weights / "yunet.onnx"
                transcript = readings.get(key, {}).get("spoken")
                audition = speech / ".auditions" / (key + ".mp4") if fallback else None
                if network is None:
                    network = Dubber(args.weights, args.musetalk, args.device)
                try:
                    result = prepare_take(
                        network,
                        source,
                        speech / (key + ".wav"),
                        speech / (key + "-face.mp4"),
                        detector,
                        args.syncnet,
                        start=shot.get("start", 0),
                        seconds=shot["seconds"],
                        batch=2 if args.device == "cuda-stream" else 4,
                        transcript=transcript,
                        audition=audition,
                    )
                except ValueError:
                    if fallback is None:
                        raise
                    # The alternate renderer owns the GPU in a separate runtime.
                    # Drop the first model and its reference cycles before starting it.
                    network = None
                else:
                    print(product["slug"], key, result["offset_ms"], flush=True)
                    continue
                gc.collect()
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                result = diffusion_fallback(
                    source,
                    speech,
                    key,
                    shot,
                    fallback,
                    detector,
                    args.syncnet,
                    transcript,
                    donor=audition,
                )
                print(product["slug"], key, result["offset_ms"], flush=True)


if __name__ == "__main__":
    main()
