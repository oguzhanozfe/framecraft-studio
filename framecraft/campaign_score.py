"""Generate an original instrumental bed in the local ComfyUI ACE-Step workflow."""

import argparse
import time
import uuid
from pathlib import Path

import httpx

from .core import save
from .providers import comfy


def compose(tags, target, *, seed=1975, seconds=32, bpm=104, front=False):
    ckpt = "ace_step_1.5_turbo_aio.safetensors"
    nodes = {}

    def node(i, kind, **inputs):
        nodes[str(i)] = {"class_type": kind, "inputs": inputs}

    node(1, "CheckpointLoaderSimple", ckpt_name=ckpt)
    node(
        2,
        "TextEncodeAceStepAudio1.5",
        clip=["1", 1],
        tags=tags,
        lyrics="[Instrumental]",
        seed=seed,
        bpm=bpm,
        duration=seconds,
        timesignature="4",
        language="en",
        keyscale="E minor",
        generate_audio_codes=True,
        cfg_scale=2.0,
        temperature=0.85,
        top_p=0.9,
        top_k=0,
        min_p=0.0,
    )
    node(3, "ConditioningZeroOut", conditioning=["2", 0])
    node(4, "EmptyAceStep1.5LatentAudio", seconds=seconds, batch_size=1)
    node(5, "ModelSamplingAuraFlow", model=["1", 0], shift=3.0)
    node(
        6,
        "KSampler",
        model=["5", 0],
        seed=seed,
        steps=8,
        cfg=1.0,
        sampler_name="euler",
        scheduler="simple",
        positive=["2", 0],
        negative=["3", 0],
        latent_image=["4", 0],
        denoise=1.0,
    )
    node(7, "VAEDecodeAudio", samples=["6", 0], vae=["1", 2])
    token = uuid.uuid4().hex
    node(
        8,
        "SaveAudioAdvanced",
        audio=["7", 0],
        filename_prefix="audio/framecraft-" + token,
        format="flac",
    )
    with httpx.Client(base_url=comfy(), timeout=60) as client:
        res = client.post(
            "/prompt", json={"prompt": nodes, "client_id": token, "front": front}
        )
        if res.is_error:
            raise RuntimeError(res.text[:1600])
        pid = res.json()["prompt_id"]
        # A soundtrack can queue behind several long video generations on one GPU.
        deadline = time.monotonic() + 3600
        while True:
            data = client.get("/history/" + pid).raise_for_status().json().get(pid)
            if data:
                if data.get("status", {}).get("status_str") == "error":
                    raise RuntimeError(
                        str(
                            [
                                entry[1].get("exception_message")
                                for entry in data["status"]["messages"]
                                if entry[0] == "execution_error"
                            ]
                        )
                    )
                record = data["outputs"]["8"]["audio"][0]
                response = client.get(
                    "/view",
                    params={k: record[k] for k in ("filename", "type", "subfolder")},
                ).raise_for_status()
                temporary = target.with_name(target.name + "." + token + ".partial")
                temporary.write_bytes(response.content)
                temporary.replace(target)
                save(
                    target.with_suffix(".json"),
                    {
                        "model": ckpt,
                        "seed": seed,
                        "prompt": tags,
                        "seconds": seconds,
                        "bpm": bpm,
                        "prompt_id": pid,
                    },
                )
                return target
            if time.monotonic() >= deadline:
                pending = (
                    client.get("/queue").raise_for_status().json()["queue_pending"]
                )
                if not any(row[1] == pid for row in pending):
                    raise TimeoutError(f"ACE-Step render exceeded one hour: {pid}")
                deadline = time.monotonic() + 3600
            time.sleep(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("direction")
    args = parser.parse_args()
    compose(args.direction, args.output)
