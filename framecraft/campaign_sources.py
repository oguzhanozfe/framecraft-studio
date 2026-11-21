"""Collect actual listing photography and developer-supplied App Store previews."""

import hashlib
import html
import io
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup
from PIL import Image, ImageOps

from .core import read, save


def fetch_models(root, voices):
    revisions = {
        "fettah": "2ebce1b57d884634c3b349670a3971f0ec9440f1",
        "fahrettin": "ec73b891c2f4f9b14b1f14dab62b90ea0eb983ad",
    }
    assets = {
        root
        / "models"
        / "depth-small.onnx": "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/4472b7362082ad9968fee890ca0f1e5aca36b93d/onnx/model_quantized.onnx"
    }
    for voice in sorted(set(voices)):
        prefix = f"tr_TR-{voice}-medium"
        for suffix in (".onnx", ".onnx.json"):
            name = prefix + suffix
            dest = root / "voices" / name
            if voice == "ilkiz":
                assets[dest] = (
                    "https://huggingface.co/ilkeraktuna/piper-tts-tr-ilkiz-medium/resolve/ecf42bcc271974d000542f5868d81bbcf4e4ec0f/ilkiz"
                    + suffix
                )
                continue
            if voice not in revisions:
                if not dest.is_file():
                    raise ValueError(f"Provide local Piper voice at {dest}")
                continue
            assets[dest] = (
                f"https://huggingface.co/csukuangfj/vits-piper-{prefix}/resolve/{revisions[voice]}/{name}"
            )
    for target, url in assets.items():
        if target.is_file():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_suffix(target.suffix + ".partial")
        with httpx.stream("GET", url, follow_redirects=True, timeout=90) as data:
            data.raise_for_status()
            with part.open("wb") as handle:
                for chunk in data.iter_bytes():
                    handle.write(chunk)
        part.replace(target)


def download_preview(client, movie, target):
    master = client.get(movie).raise_for_status().text
    renditions = re.findall(r"#EXT-X-STREAM-INF:([^\n]+)\n([^\n]+)", master)
    media = movie
    if renditions:
        _, address = max(
            renditions, key=lambda r: int(re.search(r"BANDWIDTH=(\d+)", r[0])[1])
        )
        media = urljoin(movie, address.strip())
    playlist = client.get(media).raise_for_status().text
    parts = {
        line.strip()
        for line in playlist.splitlines()
        if line.strip() and not line.startswith("#")
    }
    if len(parts) == 1 and "#EXT-X-BYTERANGE" in playlist:
        target.write_bytes(
            client.get(urljoin(media, parts.pop())).raise_for_status().content
        )
    else:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-i",
                media,
                "-map",
                "0:v:0",
                "-an",
                "-c:v",
                "copy",
                str(target),
            ],
            check=True,
            capture_output=True,
            timeout=150,
        )
    duration = float(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
                str(target),
            ]
        )
    )
    expected = sum(map(float, re.findall(r"#EXTINF:([\d.]+)", playlist)))
    if duration < expected - 0.5:
        raise ValueError("Incomplete publisher preview")


def collect(item, folder):
    folder.mkdir(parents=True, exist_ok=True)
    info_file = folder / "source.json"
    if info_file.is_file():
        return read(info_file)
    with httpx.Client(
        follow_redirects=True,
        timeout=45,
        headers={"User-Agent": "Mozilla/5.0", "Accept-Language": "tr-TR,tr;q=0.9"},
    ) as client:
        movie = ""
        if item["kind"] == "product":
            url = "https://www.amazon.com.tr/dp/" + item["id"]
            page = client.get(url).raise_for_status().text
            dom = BeautifulSoup(page, "html.parser")
            title = dom.select_one("#productTitle").get_text(" ", strip=True)
            facts = [
                n.get_text(" ", strip=True)
                for n in dom.select(
                    "#feature-bullets li .a-list-item, #pqv-feature-bullets li .a-list-item"
                )
            ]
            gallery = re.search(
                r"""'colorImages'\s*:\s*\{\s*'initial'\s*:\s*A\.\$\.parseJSON\('(.*?)'\)""",
                page,
                re.S,
            )
            urls = []
            if gallery:
                for row in json.loads(gallery[1].replace("\\'", "'")):
                    urls.append(row.get("hiRes") or row.get("large"))
            if not urls:
                lead = dom.select_one("#landingImage, #imgBlkFront")
                urls = [lead.get("data-old-hires") or lead["src"]]
        else:
            app = (
                client.get(
                    "https://itunes.apple.com/lookup",
                    params={"country": "tr", "id": item["id"]},
                )
                .raise_for_status()
                .json()["results"][0]
            )
            title, url, facts = (
                app["trackName"],
                app["trackViewUrl"],
                [app["description"][:3200]],
            )
            urls = app["screenshotUrls"] or app["ipadScreenshotUrls"]
            page = client.get(url).raise_for_status().text
            matches = re.findall(
                r'https://apptrailers\.itunes\.apple\.com/[^"<>\s]+?_default\.m3u8',
                page,
            )
            if matches:
                movie = html.unescape(matches[0])
        photos = []
        for src in list(dict.fromkeys(filter(None, urls)))[:9]:
            data = client.get(src).raise_for_status().content
            with Image.open(io.BytesIO(data)) as src_image:
                image = ImageOps.exif_transpose(src_image).convert("RGB")
                name = f"src-{len(photos) + 1:02}.jpg"
                image.save(folder / name, quality=96)
            photos.append(
                {"file": name, "url": src, "sha256": hashlib.sha256(data).hexdigest()}
            )
        if movie:
            download_preview(client, movie, folder / "preview.mp4")
        if not photos:
            raise ValueError(f"No source images for {title}")
        result = {
            "id": item["id"],
            "title": title,
            "url": url,
            "facts": list(dict.fromkeys(facts))[:8],
            "photos": photos,
            "preview_url": movie,
        }
        save(info_file, result)
        return result


def collect_all(root):
    root = Path(root)
    items = read(root / "campaign.json")["items"]
    with ThreadPoolExecutor(max_workers=4) as pool:
        tasks = [
            pool.submit(collect, item, root / item["slug"] / "source") for item in items
        ]
        for task in tasks:
            r = task.result()
            print(
                r["id"],
                r["title"],
                len(r["photos"]),
                bool(r["preview_url"]),
                flush=True,
            )
