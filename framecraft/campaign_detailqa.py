"""Check a reconstructed product detail against its exact photographed region."""

import argparse
from pathlib import Path
import gzip
import json

import cv2
import numpy as np

from .campaign_edit import fingerprint
from .core import read, save
from .director import ask
from .motion_review import Inspection


def resized(im, maximum=500):
    h, w = im.shape[:2]
    scale = min(1, maximum / max(h, w))
    return cv2.resize(
        im,
        (max(1, round(w * scale)), max(1, round(h * scale))),
        interpolation=cv2.INTER_AREA,
    )


def check_region(video, mesh_receipt, report, *, facts=(), model=None):
    initial_video = fingerprint(video)
    receipt = read(mesh_receipt)
    source = Path(receipt["source"])
    if (
        receipt["kind"] != "visible_surface_relief"
        or fingerprint(source) != receipt["source_sha256"]
    ):
        raise ValueError("Detail provenance does not match the catalog photograph")
    raw = cv2.imread(str(source))
    x, y, r, b = receipt["pixel_crop"]
    if raw is None:
        raise ValueError("Unreadable catalog photograph")
    if not (0 <= x < r <= raw.shape[1] and 0 <= y < b <= raw.shape[0]):
        raise ValueError("Invalid photographed region")
    actual = raw[y:b, x:r]
    reference = resized(actual)
    if reference.ndim != 3:
        raise ValueError("Unreadable reference detail")
    gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    sift = cv2.SIFT_create(nfeatures=900)
    pts, desc = sift.detectAndCompute(gray, None)
    if desc is None or len(desc) < 20:
        raise ValueError("Too few distinct physical details in the reference")
    matcher = cv2.BFMatcher()
    capture = cv2.VideoCapture(str(video))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    picked = {round((total - 1) * f) for f in (0.15, 0.5, 0.84)}
    strips = []
    counts = []
    passed = 0
    ledger = report.with_suffix(".frames.jsonl.gz")
    try:
        with gzip.open(ledger, "wt", encoding="utf8") as log:
            for i in range(total):
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f"Unreadable frame {i}")
                if i in picked:
                    strips.append(frame)
                small = resized(frame)
                kp, fd = sift.detectAndCompute(
                    cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), None
                )
                good = (
                    []
                    if fd is None or len(fd) < 2
                    else [
                        m
                        for pair in matcher.knnMatch(desc, fd, k=2)
                        if len(pair) == 2
                        for m, n in [pair]
                        if m.distance < 0.72 * n.distance
                    ]
                )
                coverage = 0.0
                cells = 0
                count = 0
                if len(good) >= 10:
                    a = np.float32([pts[m.queryIdx].pt for m in good])
                    z = np.float32([kp[m.trainIdx].pt for m in good])
                    transform, mask = cv2.findHomography(a, z, cv2.RANSAC, 4)
                    if transform is not None and mask is not None:
                        same = a[mask.ravel().astype(bool)]
                        count = len(same)
                        if count >= 3:
                            coverage = float(
                                cv2.contourArea(cv2.convexHull(same)) / (w * h)
                            )
                            bins = np.floor(same / [w, h] * [3, 4]).astype(int)
                            cells = len(np.unique(bins, axis=0))
                valid = count >= 18 and cells >= 4 and coverage >= 0.12
                passed += int(valid)
                counts.append(count)
                log.write(
                    json.dumps(
                        {
                            "frame": i,
                            "inliers": count,
                            "cells": cells,
                            "reference_coverage": round(coverage, 3),
                            "matched": valid,
                        }
                    )
                    + "\n"
                )
    finally:
        capture.release()
    if len(strips) != 3:
        raise ValueError("The detail shot needs three separate video observations")
    marked = raw.copy()
    cv2.rectangle(
        marked, (x, y), (r - 1, b - 1), (0, 145, 255), max(3, round(raw.shape[1] / 145))
    )

    def tile(im):
        h, w = im.shape[:2]
        scale = min(250 / w, 370 / h)
        thumb = cv2.resize(im, (max(1, round(w * scale)), max(1, round(h * scale))))
        canvas = np.full((390, 270, 3), 28, np.uint8)
        yy = (390 - thumb.shape[0]) // 2
        xx = (270 - thumb.shape[1]) // 2
        canvas[yy : yy + thumb.shape[0], xx : xx + thumb.shape[1]] = thumb
        return canvas

    board = np.hstack([tile(im) for im in [marked, actual, *strips]])
    verdict = ask(
        Inspection,
        "Inspect this physical product detail. Tile 1 is the actual seller photograph with the measured region outlined in orange. Tile 2 is the exact original region, at a comparable size. Tiles 3, 4 and 5 are successive frames of the rendered detail. Compare a visible construction in tiles 3–5 to the SAME location in tile 2. Preserve intentional openings, negative space, and separate panels that are present in the authentic item. A real new attachment, missing strap, geometric tear, or malformed visible limb is a defect. Distinguish a changed perspective from an altered construction. Name the matching or changed edges and the precise tile.",
        {"source_product_facts": list(facts)},
        cv2.imencode(".jpg", board)[1].tobytes(),
        model_override=model,
    )
    if fingerprint(video) != initial_video:
        raise ValueError("The video changed while checking the detail")
    result = {
        "passed": passed / len(counts) >= 0.95 and verdict.defect == "none",
        "video_sha256": initial_video,
        "reference_sha256": receipt["source_sha256"],
        "pixel_crop": [x, y, r, b],
        "frames": len(counts),
        "matched_frames": passed,
        "median_inliers": float(np.median(counts)),
        "ledger": ledger.name,
        "inspection": verdict.model_dump(),
    }
    save(report, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("mesh_receipt", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--model")
    parser.add_argument(
        "--source-facts", type=Path, help="Catalog JSON containing a facts array"
    )
    args = parser.parse_args()
    facts = read(args.source_facts).get("facts", []) if args.source_facts else []
    result = check_region(
        args.video, args.mesh_receipt, args.report, facts=facts, model=args.model
    )
    print(result)
    raise SystemExit(0 if result["passed"] else 2)
