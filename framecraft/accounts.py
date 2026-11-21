import hashlib
import secrets
import threading
import time
from urllib.parse import urlencode

import httpx

from . import config
from .core import setting

_pending = {}
_lock = threading.RLock()
TT_TOKEN = "https://open.tiktokapis.com/v2/oauth/token/"


def start(provider, base):
    if provider not in ("meta", "tiktok"):
        raise ValueError("Unknown account provider")
    key = setting(provider.upper() + "_CLIENT_ID")
    if not key or not setting(provider.upper() + "_CLIENT_SECRET"):
        raise ValueError("Save your application ID and secret first")
    state = secrets.token_urlsafe(32)
    verifier = secrets.token_urlsafe(48)
    redirect = base.rstrip("/") + "/auth/" + provider + "/callback"
    with _lock:
        now = time.time()
        for old in [k for k, v in _pending.items() if v[0] < now]:
            _pending.pop(old)
        _pending[state] = (now + 600, provider, redirect, verifier)
    params = {"state": state, "redirect_uri": redirect, "response_type": "code"}
    if provider == "tiktok":
        url = "https://www.tiktok.com/v2/auth/authorize/"
        params.update(
            client_key=key,
            scope="user.info.basic,video.publish",
            code_challenge=hashlib.sha256(verifier.encode()).hexdigest(),
            code_challenge_method="S256",
        )
    else:
        url = (
            "https://www.facebook.com/"
            + setting("META_VERSION", "v27.0")
            + "/dialog/oauth"
        )
        params.update(
            client_id=key,
            scope="pages_show_list,pages_read_engagement,pages_manage_posts,ads_read,ads_management",
        )
    return url + "?" + urlencode(params), state


def tt_credentials(data):
    if "access_token" not in data:
        raise ValueError("TikTok authorization failed")
    values = {
        "TIKTOK_TOKEN": data["access_token"],
        "TIKTOK_EXPIRES": str(time.time() + float(data["expires_in"])),
    }
    if data.get("refresh_token"):
        values["TIKTOK_REFRESH"] = data["refresh_token"]
    config.configure(values)


def tiktok_token():
    with _lock:
        expiry = setting("TIKTOK_EXPIRES")
        if expiry and float(expiry) < time.time() + 90 and setting("TIKTOK_REFRESH"):
            response = httpx.post(
                TT_TOKEN,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": setting("TIKTOK_REFRESH"),
                    "client_key": setting("TIKTOK_CLIENT_ID"),
                    "client_secret": setting("TIKTOK_CLIENT_SECRET"),
                },
                timeout=30,
            )
            tt_credentials(response.raise_for_status().json())
        return setting("TIKTOK_TOKEN")


def finish(provider, state, cookie, code):
    if not cookie or not secrets.compare_digest(cookie, state):
        raise ValueError("Authorization state mismatch")
    with _lock:
        item = _pending.pop(state, None)
    if not item or item[0] < time.time() or item[1] != provider:
        raise ValueError("Authorization expired")
    _, _, redirect, verifier = item
    if provider == "tiktok":
        result = httpx.post(
            TT_TOKEN,
            data={
                "code": code,
                "code_verifier": verifier,
                "grant_type": "authorization_code",
                "redirect_uri": redirect,
                "client_key": setting("TIKTOK_CLIENT_ID"),
                "client_secret": setting("TIKTOK_CLIENT_SECRET"),
            },
            timeout=30,
        )
        tt_credentials(result.raise_for_status().json())
    else:
        base = "https://graph.facebook.com/" + setting("META_VERSION", "v27.0")
        r = httpx.get(
            base + "/oauth/access_token",
            params={
                "client_id": setting("META_CLIENT_ID"),
                "client_secret": setting("META_CLIENT_SECRET"),
                "code": code,
                "redirect_uri": redirect,
            },
            timeout=30,
        )
        config.configure({"META_TOKEN": r.raise_for_status().json()["access_token"]})


def pages():
    base = "https://graph.facebook.com/" + setting("META_VERSION", "v27.0")
    url = base + "/me/accounts"
    headers = {"Authorization": "Bearer " + setting("META_TOKEN")}
    params = {"fields": "id,name,access_token", "limit": "100"}
    out = []
    while True:
        data = (
            httpx.get(url, params=params, headers=headers, timeout=30)
            .raise_for_status()
            .json()
        )
        out.extend(data["data"])
        paging = data.get("paging", {})
        if not paging.get("next"):
            return out
        params["after"] = paging["cursors"]["after"]


def select_page(page_id):
    page = next((p for p in pages() if p["id"] == page_id), None)
    if not page:
        raise ValueError("Page unavailable")
    config.configure({"META_PAGE": page_id, "META_PAGE_TOKEN": page["access_token"]})
    return {"name": page["name"]}
