import json
import os
import sqlite3
import time
import uuid
from contextlib import closing
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


def home():
    path = Path(os.environ.get("FRAMECRAFT_HOME", Path.home() / ".framecraft"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def setting(key, default=""):
    return os.environ.get("FRAMECRAFT_" + key) or default


class Brief(BaseModel):
    brief: str = Field(min_length=3, max_length=3000)
    platform: Literal["vertical", "square", "wide"] = "vertical"
    language: Literal["tr", "en"] = "tr"
    variants: int = Field(2, ge=1, le=10)
    shots: int = Field(3, ge=1, le=6)
    provider: Literal["auto", "h3", "ltx", "standalone"] = "auto"
    reference: str | None = Field(None, pattern="^[a-f0-9]{32}$")
    use_footage: bool = False
    hook: str = Field("", max_length=70)
    cta: str = Field("", max_length=70)


def db():
    con = sqlite3.connect(home() / "studio.sqlite", timeout=30)
    con.row_factory = sqlite3.Row
    con.executescript("""
    CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, kind TEXT, status TEXT, detail TEXT, updated REAL);
    CREATE TABLE IF NOT EXISTS takes(job TEXT, variant INTEGER, provider TEXT, angle TEXT, score REAL, passed INTEGER, seconds REAL, PRIMARY KEY(job,variant));
    CREATE TABLE IF NOT EXISTS outcomes(job TEXT, variant INTEGER, impressions INTEGER, clicks INTEGER, conversions INTEGER, PRIMARY KEY(job,variant));
    CREATE TABLE IF NOT EXISTS deliveries(job TEXT, channel TEXT, status TEXT, remote_id TEXT, PRIMARY KEY(job,channel));
    """)
    return con


def query(sql, args=()):
    with closing(db()) as con:
        with con:
            return [dict(row) for row in con.execute(sql, args)]


def record(job, status, detail):
    query(
        "UPDATE jobs SET status=?, detail=?,updated=? WHERE id=?",
        (status, detail, time.time(), job),
    )


def jobdir(job):
    if len(job) != 32 or any(c not in "0123456789abcdef" for c in job):
        raise ValueError("Invalid job id")
    return home() / "jobs" / job


def create_job(kind):
    job = uuid.uuid4().hex
    jobdir(job).mkdir(parents=True)
    query("INSERT INTO jobs VALUES(?,?,?,?,?)", (job, kind, "queued", "", time.time()))
    return job


def save(path, data):
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))
