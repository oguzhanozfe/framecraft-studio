import json
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import httpx

from .core import query, setting
from .preview import PreviewMonitor

GRAPHS = Path(__file__).with_name("graphs")
SIZES = {"vertical": (448, 800), "square": (576, 576), "wide": (800, 448)}
MODELS = {
    "dit": "minimax_h3_fl2va_pruned-UD-Q2_K_XL.gguf",
    "llm": "qwen3vl_32b_minimax_h3-Q2_K_M.gguf",
    "vae": "minimax_h3_video_vae_fp16.safetensors",
    "audio": "minimax_h3_audio_vae_fp32.safetensors",
}


def comfy():
    return setting("COMFY_URL", "http://127.0.0.1:8188").rstrip("/")


def template(name):
    return json.loads((GRAPHS / (name + ".json")).read_text(encoding="utf-8-sig"))


def standalone_paths():
    root = Path(setting("SD_HOME", "."))
    exe = next(
        (
            p
            for p in (
                root / "runtime" / "sd-cli.exe",
                root / "sd-cli",
                root / "sd-cli.exe",
            )
            if p.is_file()
        ),
        None,
    )
    return exe, {key: root / "models" / name for key, name in MODELS.items()}


def missing(graph, info):
    reasons = []
    for n in graph.values():
        cls = n["class_type"]
        if cls not in info:
            reasons.append(cls)
            continue
        fields = info[cls]["input"].get("required", {})
        for key, value in n["inputs"].items():
            if not key.endswith("_name") or not isinstance(value, str):
                continue
            entry = fields.get(key, [])
            if entry and isinstance(entry[0], list) and value not in entry[0]:
                reasons.append(value)
    return sorted(set(reasons))


def inventory():
    try:
        info = httpx.get(comfy() + "/object_info", timeout=8).raise_for_status().json()
        devices = (
            httpx.get(comfy() + "/system_stats", timeout=5)
            .raise_for_status()
            .json()["devices"]
        )
    except (httpx.HTTPError, KeyError, ValueError):
        info, devices = {}, []
    gpu = next((d for d in devices if "cuda" in d["name"].lower()), None)
    if not gpu and shutil.which("nvidia-smi"):
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
        )
        if out.returncode == 0 and out.stdout.strip():
            name, size = out.stdout.splitlines()[0].rsplit(",", 1)
            gpu = {"name": name, "vram_total": int(size.strip()) * 1024**2}
    gb = round(gpu["vram_total"] / 1024**3, 1) if gpu else None
    stats = {
        row["provider"]: row
        for row in query(
            """SELECT t.provider,count(*) tests,avg(coalesce(r.score,t.score)) quality,
                      avg(coalesce(r.passed,t.passed)) pass_rate,avg(t.seconds) seconds,
                      count(r.comparison) comparative_reviews
               FROM takes t LEFT JOIN model_reviews r ON r.job=t.job AND r.variant=t.variant
               GROUP BY t.provider"""
        )
    }
    choices = []
    for name in ("h3", "ltx", "wan", "standalone"):
        if name == "standalone":
            exe, paths = standalone_paths()
            absent = [p.name for p in paths.values() if not p.is_file()]
            if not exe:
                absent.append("sd-cli")
        else:
            absent = missing(template(name), info) if info else ["ComfyUI offline"]
        recommended = {"standalone": 12, "wan": 8}.get(name, 24)
        fit = (gb or 0) >= recommended
        measured = stats.get(name, {})
        prior = {"h3": 82, "ltx": 78, "wan": 76, "standalone": 73}[name]
        rating = (
            prior
            if not measured
            else (prior * 2 + measured["quality"] * measured["tests"])
            / (2 + measured["tests"])
        )
        if measured:
            rating -= (
                12
                * (1 - measured["pass_rate"])
                * measured["tests"]
                / (measured["tests"] + 2)
            )
        if not fit:
            rating -= 15
        choices.append(
            dict(
                id=name,
                available=not absent,
                missing=absent,
                planning_vram_gib=recommended,
                headroom_estimate="tested class"
                if fit
                else "offload or smaller canvas advised",
                rank=round(rating, 2),
                prior=prior,
                history_delta=round(rating + (0 if fit else 15) - prior, 2),
                hardware_delta=0 if fit else -15,
                history=measured,
            )
        )
    return {
        "gpu": gpu["name"] if gpu else None,
        "vram_gib": gb,
        "models": sorted(choices, key=lambda c: -c["rank"]),
    }


def generate(
    name,
    prompt,
    seed,
    shape,
    target,
    progress,
    lora=None,
    *,
    frames=73,
    reference=None,
    end_reference=None,
    front=False,
):
    if end_reference is not None and name != "h3":
        raise ValueError("Terminal-frame conditioning requires the H3 graph")
    if reference is not None and name not in {"h3", "ltx", "wan"}:
        raise ValueError(
            "Reference-conditioned campaign shots require H3, LTX or Wan in ComfyUI"
        )
    width, height = SIZES[shape]
    if name == "standalone":
        exe, paths = standalone_paths()
        out = target.with_suffix(".webm")
        cmd = [
            str(exe),
            "--mode",
            "vid_gen",
            "--diffusion-model",
            str(paths["dit"]),
            "--llm",
            str(paths["llm"]),
            "--vae",
            str(paths["vae"]),
            "--audio-vae",
            str(paths["audio"]),
            "--prompt",
            prompt,
            "--width",
            str(width),
            "--height",
            str(height),
            "--video-frames",
            str(frames),
            "--fps",
            "24",
            "--steps",
            "8",
            "--cfg-scale",
            "1",
            "--seed",
            str(seed),
            "--backend",
            "te=cpu",
            "--offload-to-cpu",
            "--flash-attn",
            "off,diffusion=on",
            "--output",
            str(out),
        ]
        with target.with_suffix(".log").open("w") as log:
            subprocess.run(cmd, stdout=log, stderr=log, check=True, timeout=3600)
        return out
    g = template(name)
    if name == "h3":
        g["6"]["inputs"].update(
            prompt=prompt, width=width, height=height, length=frames
        )
        g["10"]["inputs"]["noise_seed"] = seed
        if lora:
            g["16"] = {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {
                    "model": ["5", 0],
                    "lora_name": lora,
                    "strength_model": 0.55,
                },
            }
            for key in ("7", "8"):
                g[key]["inputs"]["model"] = ["16", 0]
    elif name == "wan":
        g["4"]["inputs"]["text"] = prompt
        g["6"]["inputs"].update(width=width, height=height, length=frames)
        g["8"]["inputs"]["seed"] = seed
    else:
        g["2"]["inputs"]["text"] = prompt
        g["3"]["inputs"]["text"] = (
            "unreadable lettering, deformed hands, unstable objects, overlay, watermark"
        )
        g["5"]["inputs"].update(width=width, height=height, length=frames)
        g["6"]["inputs"]["seed"] = seed
    output = {"h3": "15", "ltx": "8", "wan": "11"}[name]
    token = uuid.uuid4().hex
    g[output]["inputs"]["filename_prefix"] = "video/framecraft-" + token
    with httpx.Client(base_url=comfy(), timeout=35) as client:
        for slot, frame in (("17", reference), ("19", end_reference)):
            if frame is None:
                continue
            with frame.open("rb") as image:
                upload = (
                    client.post(
                        "/upload/image",
                        files={
                            "image": (
                                "framecraft-" + token + "-" + slot + frame.suffix,
                                image,
                            )
                        },
                        data={"type": "input"},
                    )
                    .raise_for_status()
                    .json()
                )
            name_in_comfy = "/".join(
                filter(None, (upload.get("subfolder"), upload["name"]))
            )
            g[slot] = {"class_type": "LoadImage", "inputs": {"image": name_in_comfy}}
            if name == "h3":
                g["6"]["inputs"]["first_frame" if slot == "17" else "last_frame"] = [
                    slot,
                    0,
                ]
            elif name == "wan":
                g["6"]["inputs"]["start_image"] = ["17", 0]
            else:
                # The LTX 2.5 guide node returns a stage-one lattice at half the requested dimensions.
                g["18"] = {
                    "class_type": "LTXV25ImgToVideo",
                    "inputs": {
                        "model": ["1", 0],
                        "clip": ["1", 1],
                        "vae": ["1", 2],
                        "audio_vae": ["1", 3],
                        "mode": "i2v",
                        "images": ["17", 0],
                        "prompt": prompt,
                        "negative_prompt": g["3"]["inputs"]["text"],
                        "width": 2 * width,
                        "height": 2 * height,
                        "frame_rate": 24.0,
                        "length": frames,
                        "batch_size": 1,
                        "image_strength": 0.85,
                        "img_compression": 18,
                    },
                }
                g["6"]["inputs"].update(
                    model=["18", 0],
                    positive=["18", 1],
                    negative=["18", 2],
                    latent_image=["18", 3],
                )
        monitor = PreviewMonitor(comfy(), token, target.parent)
        try:
            request = {
                "prompt": g,
                "front": front,
                "client_id": token,
                "extra_data": {"preview_method": "latent2rgb"},
            }
            resp = client.post("/prompt", json=request)
            if resp.is_error:
                raise RuntimeError("ComfyUI rejected graph: " + resp.text[:900])
            pid = resp.json()["prompt_id"]
            deadline = time.monotonic() + 3600
            progress(f"{name} · render {pid}")
            next_history = 0.0
            retried_read = False
            while time.monotonic() < deadline:
                monitor.inspect(client, pid, progress)
                if time.monotonic() < next_history:
                    continue
                next_history = time.monotonic() + 2
                data = client.get("/history/" + pid).raise_for_status().json().get(pid)
                if data:
                    if data.get("status", {}).get("status_str") == "error":
                        errors = [
                            msg.get("exception_message", "")
                            for event, msg in data["status"]["messages"]
                            if event == "execution_error"
                        ]
                        if not retried_read and any(
                            "HostBuffer.read_file_slice failed" in e for e in errors
                        ):
                            retried_read = True
                            pid = (
                                client.post("/prompt", json=request)
                                .raise_for_status()
                                .json()["prompt_id"]
                            )
                            progress(f"Retrying model weight read · {pid}")
                            continue
                        raise RuntimeError("ComfyUI: " + " ".join(errors)[:800])
                    items = data.get("outputs", {}).get(output, {}).get("images", [])
                    if items:
                        params = {
                            k: items[0][k] for k in ("filename", "subfolder", "type")
                        }
                        with (
                            client.stream("GET", "/view", params=params) as result,
                            target.open("wb") as f,
                        ):
                            result.raise_for_status()
                            for chunk in result.iter_bytes():
                                f.write(chunk)
                        return target
            raise TimeoutError("ComfyUI render exceeded one hour: " + pid)
        finally:
            monitor.close()
