"""Produce a thirty-second campaign from a supported store listing."""

import re
import shutil
from urllib.parse import urlsplit
from typing import Literal

import av
from pydantic import BaseModel, Field, field_validator
from .core import jobdir, read, record, save


def listing(url):
    page = urlsplit(url)
    if page.scheme != "https" or page.username or page.password or page.port:
        raise ValueError("Use an HTTPS Amazon Türkiye or App Store listing")
    if page.hostname in {"amazon.com.tr", "www.amazon.com.tr"}:
        product = re.search(r"/(?:dp|gp/product)/([A-Z0-9]{10})(?:/|$)", page.path)
        if product:
            return "product", product[1]
    if page.hostname == "apps.apple.com":
        game = re.search(r"/id(\d+)(?:/|$)", page.path)
        if game:
            return "game", game[1]
    raise ValueError(
        "Amazon Türkiye ürün bağlantısı veya App Store oyun bağlantısı gerekli."
    )


class ProductLink(BaseModel):
    url: str = Field(min_length=12, max_length=10000)
    project: str = Field("", max_length=72)
    audience: str = Field("", max_length=400)
    angle: str = Field("", max_length=1500)
    camera: Literal["coverage", "handheld"] = "coverage"
    audio: Literal["silent", "music", "voice_music"] = "silent"
    voice_profile: Literal["freya", "pocket/female_1", "pocket/female_2"] = "freya"

    @field_validator("url")
    @classmethod
    def valid_listing(cls, url):
        listing(url)
        return url


def produce_link(job):
    from .campaign_sources import collect
    from .motion_campaign import produce

    root = jobdir(job)
    request = ProductLink.model_validate(read(root / "listing.json"))
    kind, store_id = listing(request.url)
    record(job, "running", "Ürün bilgileri ve kaynak görüntüler alınıyor")
    source = root / "source"
    evidence = collect({"kind": kind, "id": store_id}, source)
    name = evidence["title"]
    item = {
        "slug": store_id,
        "kind": kind,
        "id": store_id,
        "name": name,
        "project": request.project.strip() or name[:72],
        "audience": request.audience.strip()
        or "Choose one specific audience whose real situation fits the sourced product.",
        "angle": request.angle.strip()
        or "Build a plausible ownership desire from one documented detail. Open on a real action and demonstrate the product in that context.",
        "proof": "\n".join([name, *evidence["facts"]]),
        "camera": request.camera,
        "audio": request.audio,
        "voice_profile": request.voice_profile,
        "silent": request.audio == "silent",
    }
    if request.audio != "silent":
        item["score_direction"] = (
            "Original understated instrumental bedroom groove. Warm keys, light rhythmic percussion, space for one Turkish voice-over. No singing, vocal samples or risers."
        )
    if request.audio == "voice_music":
        from .core import setting

        item["asr_model"] = setting("TRANSCRIBE_MODEL")
    if kind == "game":
        with av.open(str(source / "preview.mp4")) as media:
            stream = media.streams.video[0]
            item["portrait_game"] = stream.height > stream.width
    save(root / "item.json", item)
    record(
        job,
        "running",
        "Yerel model hedef kitleyi, açılışı ve ürün çekimlerini planlıyor",
    )
    master = produce(item, root)
    shutil.copyfile(master, root / "winner.mp4")
    report = read(master.parent / "motion-assessment.json")
    save(
        root / "result.json",
        {
            "winner": {"variant": 1, "passed": report["passed"]},
            "project": item["project"],
            "intent": read(master.parent / "shots.json")["intent"],
            "source": evidence["url"],
        },
    )
    record(job, "complete", "30 saniyelik UGC proje klasörüne kaydedildi")
