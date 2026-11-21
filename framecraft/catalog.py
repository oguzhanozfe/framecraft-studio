import hashlib
import os
import re
import shutil
import tempfile
from pathlib import Path

import httpx

from .core import setting
from .providers import template

CIVIT = "https://civitai.com/api/v1"


def compatible(base):
    return re.sub("[^a-z0-9]", "", base.lower()) in {"minimaxh3", "minimaxh3video"}


def civit_headers():
    token = setting("CIVITAI_TOKEN")
    return {"Authorization": "Bearer " + token} if token else {}


def discover(term):
    r = (
        httpx.get(
            CIVIT + "/models",
            params={
                "query": term,
                "types": "LORA",
                "limit": 25,
                "sort": "Highest Rated",
            },
            headers=civit_headers(),
            timeout=25,
        )
        .raise_for_status()
        .json()
    )
    rows = []
    for model in r["items"]:
        for v in model.get("modelVersions", []):
            if not compatible(v.get("baseModel", "")):
                continue
            examples = [im.get("meta") or {} for im in v.get("images", [])]
            rows.append(
                {
                    "name": model["name"],
                    "version": v["id"],
                    "base": v["baseModel"],
                    "trigger_words": v.get("trainedWords", []),
                    "examples": [
                        e["prompt"][:800]
                        for e in examples
                        if isinstance(e.get("prompt"), str)
                    ][:2],
                    "commercial_use": model.get("allowCommercialUse"),
                    "source": f"https://civitai.com/models/{model['id']}",
                }
            )
            break
    return rows


def fetch(url, dest, digest, headers=None):
    if not re.fullmatch("[0-9a-fA-F]{64}", digest):
        raise ValueError("No published SHA-256")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        with dest.open("rb") as f:
            actual = hashlib.file_digest(f, "sha256").hexdigest()
        if actual == digest.lower():
            return dest
        raise ValueError("Existing file has a different checksum: " + dest.name)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=dest.parent, delete=False, suffix=".part"
        ) as f:
            tmp = Path(f.name)
            sha = hashlib.sha256()
            with httpx.stream(
                "GET", url, headers=headers, follow_redirects=True, timeout=90
            ) as response:
                response.raise_for_status()
                for block in response.iter_bytes(1024 * 1024):
                    sha.update(block)
                    f.write(block)
            if sha.hexdigest() != digest.lower():
                raise ValueError("Download checksum mismatch")
        os.replace(tmp, dest)
        return dest
    finally:
        if tmp:
            tmp.unlink(missing_ok=True)


def models_dir():
    root = setting("COMFY_MODELS")
    if not root:
        raise ValueError("Set the ComfyUI models directory in Connections first")
    return Path(root)


def install_lora(version):
    result = (
        httpx.get(
            f"{CIVIT}/model-versions/{int(version)}",
            headers=civit_headers(),
            timeout=25,
        )
        .raise_for_status()
        .json()
    )
    if not compatible(result["baseModel"]):
        raise ValueError("This adapter targets a different base model")
    model = (
        httpx.get(
            f"{CIVIT}/models/{int(result['modelId'])}",
            headers=civit_headers(),
            timeout=25,
        )
        .raise_for_status()
        .json()
    )
    if model.get("type") != "LORA":
        raise ValueError("Expected a LoRA model")
    f = next(
        (
            v
            for v in result["files"]
            if v["name"].lower().endswith(".safetensors") and v.get("type") == "Model"
        ),
        None,
    )
    if not f:
        raise ValueError("This version has no SafeTensor weights")
    url = f"https://civitai.com/api/download/models/{int(version)}"
    return fetch(
        url,
        models_dir() / "loras" / f"civitai-h3-{int(version)}.safetensors",
        f.get("hashes", {}).get("SHA256", ""),
        civit_headers(),
    )


def install_model(name):
    if name not in ("h3", "ltx", "wan"):
        raise ValueError("Unknown family")
    g = template(name)
    out = models_dir()
    if name == "h3":
        repo = "Comfy-Org/MiniMax-H3"
        paths = {
            folder + "/" + g[n]["inputs"][field]: folder
            for n, folder, field in [
                ("1", "diffusion_models", "unet_name"),
                ("2", "text_encoders", "clip_name"),
                ("3", "vae", "vae_name"),
                ("4", "vae", "vae_name"),
                ("5", "loras", "lora_name"),
            ]
        }
    elif name == "wan":
        repo = "Comfy-Org/Wan-2.2"
        paths = {
            "split_files/" + sub + "/" + g[node]["inputs"][key]: sub
            for sub, node, key in (
                ("diffusion_models", "1", "unet_name"),
                ("text_encoders", "2", "clip_name"),
                ("vae", "3", "vae_name"),
            )
        }
    else:
        repo = "ChrisColeTech/LTX-2.5-uncensored-v1.1-FP8"
        data = g["1"]["inputs"]
        paths = {
            "split/" + sub + "/" + data[k]: dest
            for k, sub, dest in [
                ("unet_name", "diffusion_models", "diffusion_models"),
                ("clip_name", "text_encoders/gguf", "text_encoders"),
                ("video_vae_name", "vae", "vae"),
                ("audio_vae_name", "vae", "vae"),
            ]
        }
        paths[
            "split/text_encoders/gguf/mmproj-gemma4_12b_ltx25_uncensored-f16.gguf"
        ] = "text_encoders"
    info = (
        httpx.get(
            "https://huggingface.co/api/models/" + repo,
            params={"blobs": "true"},
            follow_redirects=True,
            timeout=25,
        )
        .raise_for_status()
        .json()
    )
    meta = {s["rfilename"]: s for s in info["siblings"]}
    out.mkdir(parents=True, exist_ok=True)
    needed = sum(
        meta[p]["size"]
        for p, d in paths.items()
        if not (out / d / Path(p).name).exists()
    )
    if shutil.disk_usage(out).free < needed:
        raise ValueError("Insufficient disk space")
    return [
        str(
            fetch(
                f"https://huggingface.co/{repo}/resolve/{info['sha']}/{p}",
                out / d / Path(p).name,
                meta[p]["lfs"]["sha256"],
            )
        )
        for p, d in paths.items()
    ]
