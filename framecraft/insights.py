import json
from datetime import date

import httpx
from pydantic import BaseModel, Field

from .core import query, setting
from .publish import meta_base


class Attribution(BaseModel):
    ad_id: str = Field(pattern=r"^\d{1,30}$")
    variant: int = Field(ge=1, le=10)
    since: date
    until: date
    conversion_action: str = Field(default="purchase", max_length=100)


def sync(job, link):
    if link.since > link.until:
        raise ValueError("Invalid reporting window")
    if not query(
        "SELECT variant FROM takes WHERE job=? AND variant=?", (job, link.variant)
    ):
        raise ValueError("Unknown variant")
    token = setting("META_TOKEN")
    if not token:
        raise ValueError("Set a Meta token with ads_read")
    params = {
        "fields": "impressions,clicks,actions",
        "time_range": json.dumps({"since": str(link.since), "until": str(link.until)}),
    }
    res = httpx.get(
        meta_base() + "/" + link.ad_id + "/insights",
        params=params,
        headers={"Authorization": "Bearer " + token},
        timeout=40,
    )
    rows = res.raise_for_status().json()["data"]
    if not rows:
        raise ValueError("No data for this ad and date range")
    impressions = sum(int(r.get("impressions", 0)) for r in rows)
    clicks = sum(int(r.get("clicks", 0)) for r in rows)
    conversions = sum(
        float(a["value"])
        for r in rows
        for a in r.get("actions", [])
        if a["action_type"] == link.conversion_action
    )
    if impressions < 1:
        raise ValueError("No impressions in the reporting window")
    query(
        "INSERT OR REPLACE INTO outcomes VALUES(?,?,?,?,?)",
        (job, link.variant, impressions, clicks, conversions),
    )
    return {
        "source": "Meta Ads Insights",
        "ad_id": link.ad_id,
        "impressions": impressions,
        "clicks": clicks,
        "conversions": conversions,
        "window": [str(link.since), str(link.until)],
    }
