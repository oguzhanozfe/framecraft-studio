import os
import threading

import keyring

from .core import home, read, save

PUBLIC = (
    "TIKTOK_CLIENT_ID",
    "META_CLIENT_ID",
    "TIKTOK_EXPIRES",
    "COMFY_URL",
    "COMFY_MODELS",
    "SD_HOME",
    "LLM_URL",
    "LLM_MODEL",
    "VISION_MODEL",
    "DIRECTOR",
    "H3_LORA",
    "META_PAGE",
    "META_AD_ACCOUNT",
    "META_VERSION",
    "AUTO_LORA",
    "TRANSCRIBE_MODEL",
    "VOICE_PYTHON",
    "FREYA_MODEL",
    "FREYA_SOURCE",
    "VOICE_DEVICE",
)
SECRETS = (
    "TIKTOK_CLIENT_SECRET",
    "META_CLIENT_SECRET",
    "META_PAGE_TOKEN",
    "TIKTOK_REFRESH",
    "LLM_KEY",
    "CIVITAI_TOKEN",
    "TIKTOK_TOKEN",
    "META_TOKEN",
)
lock = threading.RLock()


def setup():
    file = home() / "settings.json"
    data = read(file) if file.exists() else {}
    for key in PUBLIC:
        if key in data:
            os.environ.setdefault("FRAMECRAFT_" + key, str(data[key]))
    for key in SECRETS:
        if key in data.get("saved_secrets", []):
            value = keyring.get_password("framecraft-studio", key)
            if value:
                os.environ.setdefault("FRAMECRAFT_" + key, value)


def configure(values):
    with lock:
        path = home() / "settings.json"
        data = read(path) if path.exists() else {}
        saved = set(data.get("saved_secrets", []))
        for key, value in values.items():
            if key not in PUBLIC + SECRETS or not isinstance(value, str):
                raise ValueError("Unknown setting")
            value = value.strip()
            if key in SECRETS:
                if not value:
                    continue
                keyring.set_password("framecraft-studio", key, value)
                saved.add(key)
            else:
                data[key] = value
            os.environ["FRAMECRAFT_" + key] = value
        data["saved_secrets"] = sorted(saved)
        save(path, data)
    return visible()


def visible():
    return {
        **{k: os.environ.get("FRAMECRAFT_" + k, "") for k in PUBLIC},
        **{k: bool(os.environ.get("FRAMECRAFT_" + k)) for k in SECRETS},
    }
