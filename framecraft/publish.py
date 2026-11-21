from urllib.parse import urlsplit

import httpx

from .accounts import tiktok_token
from .core import jobdir, query, setting

TIK = "https://open.tiktokapis.com/v2"
CHANNELS = ("tiktok", "facebook", "meta_asset")


def chunks(size):
    unit = min(size, 16 * 1024 * 1024)
    if unit <= 0:
        raise ValueError("Empty file")
    count = max(1, size // unit)
    if count > 1000:
        raise ValueError("Video exceeds upload size")
    return unit, count


def tt(path, body):
    res = httpx.post(
        TIK + path,
        headers={"Authorization": "Bearer " + tiktok_token()},
        json=body,
        timeout=40,
    )
    data = res.raise_for_status().json()
    if data.get("error", {}).get("code") != "ok":
        raise ValueError(str(data.get("error", {}).get("message", "TikTok API error")))
    return data["data"]


def meta_base():
    return "https://graph.facebook.com/" + setting("META_VERSION", "v27.0")


def remember(job, channel, status, remote=""):
    query(
        "UPDATE deliveries SET status=?,remote_id=? WHERE job=? AND channel=?",
        (status, str(remote), job, channel),
    )


def deliver(job, channel, options):
    if channel not in CHANNELS:
        raise ValueError("Unsupported destination")
    file = jobdir(job) / "winner.mp4"
    if not file.is_file():
        raise ValueError("Job has no passing winner")
    if not options.get("consent"):
        raise ValueError("Confirm this specific upload")
    title = options.get("title", "")
    size = file.stat().st_size
    if channel == "tiktok":
        if not setting("TIKTOK_TOKEN"):
            raise ValueError("Connect TikTok first")
        creator = tt("/post/publish/creator_info/query/", {})
        if options.get("privacy") not in creator["privacy_level_options"]:
            raise ValueError("Select an available privacy level")
        import av

        with av.open(str(file)) as video:
            if video.duration / av.time_base > creator["max_video_post_duration_sec"]:
                raise ValueError("Video exceeds creator duration limit")
        if options.get("branded") and options["privacy"] == "SELF_ONLY":
            raise ValueError("Branded content requires an eligible visibility")
    elif not setting("META_TOKEN") or not setting(
        "META_PAGE" if channel == "facebook" else "META_AD_ACCOUNT"
    ):
        raise ValueError("Set the destination ID and Meta access token")
    inserted = query(
        "INSERT OR IGNORE INTO deliveries VALUES(?,?,?,?) RETURNING job",
        (job, channel, "starting", ""),
    )
    if not inserted:
        raise ValueError("Delivery already recorded. Check its status before retrying.")
    remote = ""
    try:
        if channel == "tiktok":
            unit, count = chunks(size)
            info = {
                "privacy_level": options["privacy"],
                "title": title,
                "is_aigc": True,
                "disable_comment": creator.get("comment_disabled", False)
                or not options.get("comments", False),
                "disable_duet": creator.get("duet_disabled", False)
                or not options.get("duet", False),
                "disable_stitch": creator.get("stitch_disabled", False)
                or not options.get("stitch", False),
                "brand_content_toggle": bool(options.get("branded")),
                "brand_organic_toggle": bool(options.get("own_brand")),
            }
            data = tt(
                "/post/publish/video/init/",
                {
                    "post_info": info,
                    "source_info": {
                        "source": "FILE_UPLOAD",
                        "video_size": size,
                        "chunk_size": unit,
                        "total_chunk_count": count,
                    },
                },
            )
            remote = data["publish_id"]
            remember(job, channel, "uploading", remote)
            with file.open("rb") as stream:
                for i in range(count):
                    start = stream.tell()
                    chunk = stream.read(unit if i < count - 1 else size - start)
                    httpx.put(
                        data["upload_url"],
                        content=chunk,
                        headers={
                            "Content-Type": "video/mp4",
                            "Content-Range": f"bytes {start}-{start + len(chunk) - 1}/{size}",
                        },
                        timeout=180,
                    ).raise_for_status()
        else:
            base = meta_base()
            token = (
                (setting("META_PAGE_TOKEN") or setting("META_TOKEN"))
                if channel == "facebook"
                else setting("META_TOKEN")
            )
            auth = {"Authorization": "Bearer " + token}
            if channel == "meta_asset":
                account = setting("META_AD_ACCOUNT").removeprefix("act_")
                with file.open("rb") as f:
                    data = (
                        httpx.post(
                            base + f"/act_{account}/advideos",
                            headers=auth,
                            files={"source": ("creative.mp4", f, "video/mp4")},
                            data={"title": title},
                            timeout=300,
                        )
                        .raise_for_status()
                        .json()
                    )
                remote = data["id"]
            else:
                url = base + "/" + setting("META_PAGE") + "/video_reels"
                data = (
                    httpx.post(
                        url, headers=auth, data={"upload_phase": "start"}, timeout=30
                    )
                    .raise_for_status()
                    .json()
                )
                remote = data["video_id"]
                remember(job, channel, "uploading", remote)
                upload = data["upload_url"]
                if urlsplit(upload).hostname != "rupload.facebook.com":
                    raise ValueError("Unexpected Meta upload host")
                with file.open("rb") as f:
                    httpx.post(
                        upload,
                        headers={
                            "Authorization": "OAuth " + token,
                            "offset": "0",
                            "file_size": str(size),
                            "Content-Length": str(size),
                        },
                        content=f,
                        timeout=300,
                    ).raise_for_status()
                httpx.post(
                    url,
                    headers=auth,
                    data={
                        "upload_phase": "finish",
                        "video_id": remote,
                        "video_state": "PUBLISHED",
                        "description": title,
                    },
                    timeout=40,
                ).raise_for_status()
        remember(job, channel, "submitted", remote)
    except Exception:
        remember(job, channel, "needs_reconciliation", remote)
        raise
    return {"status": "submitted", "remote_id": remote}


def refresh(job, channel):
    row = query("SELECT * FROM deliveries WHERE job=? AND channel=?", (job, channel))[0]
    if not row["remote_id"]:
        return row
    if channel == "tiktok":
        status = tt("/post/publish/status/fetch/", {"publish_id": row["remote_id"]})[
            "status"
        ]
    else:
        data = (
            httpx.get(
                meta_base() + "/" + row["remote_id"],
                params={"fields": "status"},
                headers={
                    "Authorization": "Bearer "
                    + (
                        (setting("META_PAGE_TOKEN") or setting("META_TOKEN"))
                        if channel == "facebook"
                        else setting("META_TOKEN")
                    )
                },
                timeout=30,
            )
            .raise_for_status()
            .json()
        )
        status = data.get("status", {}).get("video_status", "processing")
    remember(job, channel, status, row["remote_id"])
    return {"status": status, "remote_id": row["remote_id"]}
