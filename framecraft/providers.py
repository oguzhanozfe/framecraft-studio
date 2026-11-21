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
            "SELECT provider,count(*) tests,avg(score) quality,avg(passed) pass_rate,avg(seconds) seconds FROM takes GROUP BY provider"
        )
    }
    choices = []
    for name in ("h3", "ltx", "standalone"):
        if name == "standalone":
            exe, paths = standalone_paths()
            absent = [p.name for p in paths.values() if not p.is_file()]
            if not exe:
                absent.append("sd-cli")
        else:
            absent = missing(template(name), info) if info else ["ComfyUI offline"]
        recommended = 12 if name == "standalone" else 24
        fit = (gb or 0) >= recommended
        measured = stats.get(name, {})
        prior = 82 if name == "h3" else (78 if name == "ltx" else 73)
        rating = (
            prior
            if not measured
            else (prior * 2 + measured["quality"] * measured["tests"])
            / (2 + measured["tests"])
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
                history=measured,
            )
        )
    return {
        "gpu": gpu["name"] if gpu else None,
        "vram_gib": gb,
        "models": sorted(choices, key=lambda c: -c["rank"]),
    }


def generate(name, prompt, seed, shape, target, progress, lora=None):
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
            "73",
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
        g["6"]["inputs"].update(prompt=prompt, width=width, height=height, length=73)
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
    else:
        g["2"]["inputs"]["text"] = prompt
        g["3"]["inputs"]["text"] = (
            "unreadable lettering, deformed hands, unstable objects, overlay, watermark"
        )
        g["5"]["inputs"].update(width=width, height=height, length=73)
        g["6"]["inputs"]["seed"] = seed
    output = "15" if name == "h3" else "8"
    token = uuid.uuid4().hex
    g[output]["inputs"]["filename_prefix"] = "video/framecraft-" + token
    with httpx.Client(base_url=comfy(), timeout=35) as client:
        monitor = PreviewMonitor(comfy(), token, target.parent)
        try:
            resp = client.post(
                "/prompt",
                json={
                    "prompt": g,
                    "client_id": token,
                    "extra_data": {"preview_method": "latent2rgb"},
                },
            )
            if resp.is_error:
                raise RuntimeError("ComfyUI rejected graph: " + resp.text[:900])
            pid = resp.json()["prompt_id"]
            deadline = time.monotonic() + 3600
            progress(f"{name} · render {pid}")
            next_history = 0.0
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
