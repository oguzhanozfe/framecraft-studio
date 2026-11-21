import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

import av
import httpx
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from filelock import FileLock
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from . import accounts, catalog, config, insights, publish
from .compare import Comparison, compare, resolve
from .core import Brief, Revision, create_job, home, jobdir, query, read, record, save
from .pipeline import analyze_job, make
from .providers import inventory

pool = ThreadPoolExecutor(max_workers=1)
static = Path(__file__).with_name("static")


def worker(job):
    rows = query(
        "UPDATE jobs SET status='running' WHERE id=? AND status='queued' RETURNING kind",
        (job,),
    )
    if not rows:
        return
    kind = rows[0]["kind"]
    (jobdir(job) / ".worker").touch()
    try:
        if kind == "create":
            make(job)
        elif kind == "compare":
            compare(job)
        elif kind == "reference":
            analyze_job(job)
        else:
            spec = read(jobdir(job) / "install.json")
            files = (
                catalog.install_model(spec["model"])
                if "model" in spec
                else [str(catalog.install_lora(spec["version"]))]
            )
            if "version" in spec:
                config.configure({"H3_LORA": Path(files[0]).name})
            save(jobdir(job) / "result.json", {"installed": files})
            record(job, "complete", "Installed and verified")
    except Exception as exc:
        record(job, "failed", str(exc)[:900])
    finally:
        (jobdir(job) / ".worker").unlink(missing_ok=True)


@asynccontextmanager
async def lifespan(app):
    config.setup()
    with FileLock(str(home() / "worker.lock"), timeout=0):
        for row in query(
            "SELECT id,status FROM jobs WHERE status IN ('running','queued')"
        ):
            if row["status"] == "queued":
                pool.submit(worker, row["id"])
            elif (jobdir(row["id"]) / ".worker").exists():
                record(
                    row["id"],
                    "interrupted",
                    "Interrupted on restart. Retry as a new job.",
                )
        yield


app = FastAPI(title="Framecraft Studio", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["127.0.0.1", "localhost", "[::1]", "testserver"],
)


@app.middleware("http")
async def local_writes(request, call_next):
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        origin = request.headers.get("origin")
        if request.headers.get("x-framecraft") != "1" or (
            origin and urlsplit(origin).netloc != request.headers.get("host")
        ):
            return JSONResponse(
                {"detail": "Use the local Studio or send X-Framecraft: 1"},
                status_code=403,
            )
    return await call_next(request)


@app.exception_handler(ValueError)
def value_error(req, exc):
    return JSONResponse({"detail": str(exc)[:600]}, status_code=400)


@app.exception_handler(httpx.HTTPError)
def remote_error(req, exc):
    status = (
        exc.response.status_code
        if isinstance(exc, httpx.HTTPStatusError)
        else type(exc).__name__
    )
    return JSONResponse(
        {
            "detail": f"Provider request failed ({status}). Verify connection and permissions."
        },
        status_code=502,
    )


@app.get("/")
def index():
    return FileResponse(static / "index.html")


@app.get("/studio.{ext}")
def assets(ext: str):
    if ext not in ("js", "css"):
        raise HTTPException(404)
    return FileResponse(static / ("studio." + ext))


@app.get("/api/models")
def models():
    return inventory()


@app.get("/api/settings")
def settings():
    return config.visible()


@app.post("/api/settings")
def update_settings(values: dict[str, str]):
    return config.configure(values)


def submit(kind, file, data, *, retry_source=None):
    job = create_job(kind)
    save(jobdir(job) / file, data)
    if retry_source:
        save(jobdir(job) / "retry-source.json", {"job": retry_source})
    pool.submit(worker, job)
    return {"id": job}


@app.post("/api/create", status_code=202)
def create(brief: Brief):
    if brief.revision:
        revision_source(brief.revision.job, brief.revision.variant)
        if brief.use_footage:
            raise ValueError("A revision renders new footage")
    if brief.use_footage and not brief.reference:
        raise ValueError("Choose a reference video first")
    if brief.reference:
        folder = jobdir(brief.reference)
        if not (folder / "result.json").exists() or "analysis" not in read(
            folder / "result.json"
        ):
            raise ValueError("Choose a completed reference analysis")
    return submit("create", "brief.json", brief.model_dump())


class RevisionRequest(BaseModel):
    changes: str = Field(min_length=3, max_length=1500)
    variant: int | None = Field(None, ge=1, le=10)
    variants: int | None = Field(None, ge=1, le=10)


def revision_source(job, variant):
    folder = jobdir(job)
    rows = query("SELECT kind,status FROM jobs WHERE id=?", (job,))
    if not rows:
        raise HTTPException(404)
    if rows[0]["kind"] != "create" or rows[0]["status"] not in (
        "complete",
        "needs_review",
    ):
        raise ValueError("Choose a finished creative job")
    if variant is None:
        result = read(folder / "result.json")
        variant = (result.get("winner") or {}).get("variant")
    source = folder / f"v{variant}"
    if not (source / "master.mp4").is_file() or not (source / "story.json").is_file():
        raise ValueError("Choose a rendered variant with a storyboard")
    return variant


@app.post("/api/jobs/{job}/revise", status_code=202)
def revise(job: str, request: RevisionRequest):
    variant = revision_source(job, request.variant)
    brief = Brief.model_validate(read(jobdir(job) / "brief.json"))
    brief.revision = Revision(job=job, variant=variant, changes=request.changes.strip())
    brief.variants = request.variants or brief.variants
    brief.use_footage = False
    brief.reference = None
    brief.hook = brief.cta = ""
    return submit("create", "brief.json", brief.model_dump())


@app.post("/api/analyze", status_code=202)
def analyze(video: UploadFile = File(...)):
    job = create_job("reference")
    path = jobdir(job) / "input.mp4"
    try:
        with path.open("wb") as out:
            while chunk := video.file.read(1024 * 1024):
                if out.tell() + len(chunk) > 1024**3:
                    raise ValueError("Maximum upload size is 1 GiB")
                out.write(chunk)
        with av.open(
            str(path),
            options={
                "protocol_whitelist": "file",
                "format_whitelist": "mov,matroska,webm,avi",
            },
        ) as media:
            if not media.streams.video:
                raise ValueError("Missing video stream")
    except Exception:
        path.unlink(missing_ok=True)
        record(job, "failed", "Invalid or unsupported video")
        raise HTTPException(
            422, "Upload an MP4, MOV, WebM, MKV or AVI video up to 1 GiB"
        )
    pool.submit(worker, job)
    return {"id": job}


@app.get("/api/jobs")
def jobs():
    return query("SELECT * FROM jobs ORDER BY updated DESC LIMIT 100")


@app.get("/api/jobs/{job}")
def detail(job: str):
    rows = query("SELECT * FROM jobs WHERE id=?", (job,))
    if not rows:
        raise HTTPException(404)
    folder = jobdir(job)
    result = folder / "result.json"
    assets = [
        p.relative_to(folder).as_posix()
        for p in sorted(folder.rglob("*"))
        if p.is_file()
        and (
            p.suffix in (".json", ".jsonl")
            or p.name in ("master.mp4", "winner.mp4", "input.mp4")
        )
    ]
    return {
        **rows[0],
        "revision": read(folder / "brief.json").get("revision")
        if (folder / "brief.json").is_file()
        else None,
        "result": read(result) if result.exists() else None,
        "assets": assets,
        "deliveries": query("SELECT * FROM deliveries WHERE job=?", (job,)),
    }


@app.get("/api/jobs/{job}/files/{name:path}")
def artifact(job: str, name: str):
    root = jobdir(job).resolve()
    file = (root / name).resolve()
    if (
        not file.is_relative_to(root)
        or file.suffix not in (".mp4", ".json", ".jsonl", ".jpg", ".png", ".webm")
        or not file.is_file()
    ):
        raise HTTPException(404)
    return FileResponse(file)


@app.post("/api/jobs/{job}/retry", status_code=202)
def retry(job: str):
    data = detail(job)
    if data["kind"] != "create" or data["status"] not in (
        "failed",
        "needs_review",
        "interrupted",
    ):
        raise ValueError("Only a stopped creative job can be retried")
    brief = Brief.model_validate(read(jobdir(job) / "brief.json"))
    if brief.revision:
        revision_source(brief.revision.job, brief.revision.variant)
        return submit("create", "brief.json", brief.model_dump(), retry_source=job)
    return create(brief)


class Metrics(BaseModel):
    variant: int = Field(ge=1, le=10)
    impressions: int = Field(ge=1)
    clicks: int = Field(ge=0)
    conversions: int = Field(ge=0)


@app.post("/api/jobs/{job}/metrics")
def outcome(job: str, metrics: Metrics):
    if metrics.clicks > metrics.impressions or metrics.conversions > metrics.clicks:
        raise ValueError("Expected impressions >= clicks >= conversions")
    if not query(
        "SELECT 1 FROM takes WHERE job=? AND variant=?", (job, metrics.variant)
    ):
        raise ValueError("Unknown variant")
    query(
        "INSERT OR REPLACE INTO outcomes VALUES(?,?,?,?,?)",
        (
            job,
            metrics.variant,
            metrics.impressions,
            metrics.clicks,
            metrics.conversions,
        ),
    )
    return {"recorded": True}


@app.get("/api/civitai")
def civitai(q: str = "UGC"):
    return catalog.discover(q)


@app.post("/api/install/model/{name}", status_code=202)
def install(name: str):
    if name not in ("h3", "ltx"):
        raise ValueError("Unknown model")
    catalog.models_dir()
    return submit("install", "install.json", {"model": name})


@app.post("/api/install/lora/{version}", status_code=202)
def adapter(version: int):
    catalog.models_dir()
    return submit("install", "install.json", {"version": version})


@app.get("/api/tiktok/creator")
def creator():
    return publish.tt("/post/publish/creator_info/query/", {})


@app.post("/api/jobs/{job}/publish/{channel}")
def send(job: str, channel: str, options: dict):
    return publish.deliver(job, channel, options)


@app.post("/api/jobs/{job}/delivery/{channel}")
def status(job: str, channel: str):
    return publish.refresh(job, channel)


@app.post("/api/jobs/{job}/insights/meta")
def sync_metrics(job: str, attribution: insights.Attribution):
    result = insights.sync(job, attribution)
    save(jobdir(job) / f"metrics-v{attribution.variant}.json", result)
    return result


@app.get("/api/jobs/{job}/frames")
def frame_rows(
    job: str,
    report: str = "analysis",
    start: int = 0,
    limit: int = 50,
    issues_only: bool = False,
):
    root = jobdir(job).resolve()
    source = (root / report / "frames.jsonl").resolve()
    if not source.is_relative_to(root) or not source.is_file():
        raise HTTPException(404)
    if start < 0 or not 1 <= limit <= 200:
        raise ValueError("Invalid frame window")
    rows = []
    with source.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            r = json.loads(line)
            if issues_only and not r["flags"]:
                continue
            rows.append(r)
            if len(rows) >= limit:
                break
    return rows


@app.post("/api/connect/{provider}")
def connect(provider: str, request: Request):
    url, state = accounts.start(provider, str(request.base_url))
    response = JSONResponse({"authorize": url})
    response.set_cookie(
        "framecraft_oauth",
        state,
        httponly=True,
        samesite="lax",
        max_age=600,
        path="/auth/",
    )
    return response


@app.get("/auth/{provider}/callback")
def callback(provider: str, request: Request, state: str, code: str):
    accounts.finish(provider, state, request.cookies.get("framecraft_oauth"), code)
    response = RedirectResponse("/#settings")
    response.delete_cookie("framecraft_oauth", path="/auth/")
    return response


@app.get("/api/meta/pages")
def pages():
    return [{"id": p["id"], "name": p["name"]} for p in accounts.pages()]


@app.post("/api/meta/pages/{page_id}")
def select_page(page_id: str):
    return accounts.select_page(page_id)


@app.post("/api/compare", status_code=202)
def compare_candidates(spec: Comparison):
    unique = {(c.job, c.variant) for c in spec.candidates}
    if len(unique) != len(spec.candidates):
        raise ValueError("Choose different candidates")
    for candidate in spec.candidates:
        resolve(candidate)
    return submit("compare", "comparison.json", spec.model_dump())


@app.get("/api/candidates")
def candidates():
    results = []
    for job in query(
        "SELECT id,kind,updated FROM jobs WHERE status IN ('complete','needs_review') ORDER BY updated DESC LIMIT 100"
    ):
        folder = jobdir(job["id"])
        if job["kind"] == "reference" and (folder / "input.mp4").exists():
            results.append({**job, "variant": None})
        elif job["kind"] == "create":
            for row in query(
                "SELECT variant,provider,angle FROM takes WHERE job=? ORDER BY variant",
                (job["id"],),
            ):
                if (folder / f"v{row['variant']}" / "master.mp4").is_file():
                    results.append({**job, **row})
    return results
