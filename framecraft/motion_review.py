"""Inspect generated takes against their physical SKU reference before the final edit."""

from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from pydantic import BaseModel

from .campaign_edit import fingerprint
from .core import read, save
from .director import ask


class Inspection(BaseModel):
    observation: str
    defect: Literal[
        "none",
        "unrelated_product",
        "malformed_anatomy",
        "warped_geometry",
        "prominent_gibberish",
        "motion_missing",
    ]
    location: str


def reference_check(video, reference, *, model=None):
    video = Path(video)
    dest = video.with_suffix(".review.json")
    signature = (fingerprint(video), fingerprint(reference) if reference else "")
    if dest.is_file():
        cached = read(dest)
        if cached.get("input") == list(signature) and (
            cached["defect"] != "malformed_anatomy" or "anatomy_review" in cached
        ):
            return cached
    cap = cv2.VideoCapture(str(video))
    tiles = []
    try:
        for f in (0.15, 0.49, 0.82):
            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                round(max(0, cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1) * f),
            )
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Unreadable sample in {video.name}")
            tiles.append(frame)
    finally:
        cap.release()
    if reference:
        im = cv2.imread(str(reference))
        if im is None:
            raise ValueError("Unreadable product reference")
        tiles.insert(0, im)
    images = []
    for im in tiles:
        h, w = im.shape[:2]
        scale = min(300 / w, 390 / h)
        resized = cv2.resize(
            im,
            (max(1, round(w * scale)), max(1, round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
        tile = np.full((400, 310, 3), 35, np.uint8)
        y = (400 - resized.shape[0]) // 2
        x = (310 - resized.shape[1]) // 2
        tile[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
        images.append(tile)
    if len(images) == 4:
        board = np.vstack([np.hstack(images[:2]), np.hstack(images[2:])])
    else:
        board = np.hstack(images)
    picture = cv2.imencode(".jpg", board, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()
    result = ask(
        Inspection,
        "Review the actual pixels. "
        + (
            "The top-left is the source reference. The remaining three tiles show successive moments in a generated take. Compare the product's silhouette and distinguishing physical details."
            if reference
            else "The three tiles show successive moments in the same creator shot."
        )
        + " Record a severe defect only if it is clearly visible. Identify its exact location and describe the discrepancy. Ignore minor blur, camera angle changes and text that is simply too small to resolve. A hidden face or partly obscured hand is normal. Use none if no material defect is visible.",
        {"purpose": "Inspect a prospective product film take."},
        picture,
        model_override=model,
    )
    value = {"input": list(signature), **result.model_dump()}
    if result.defect == "malformed_anatomy":
        checked = ask(
            Inspection,
            "Inspect the visible anatomy and product in these frames. The first tile "
            + (
                "is the product reference."
                if reference
                else "is the first video sample."
            )
            + " Check each visible joint and limb in the remaining tiles, accounting for "
            "foreshortening, hair, garment edges and hand occlusion. Name the specific "
            "body part and physically impossible connection if one is visible. A real "
            "change of product geometry or garbled writing is a separate defect. Report "
            "the most significant defect with its tile and location. If only fabric "
            "texture, viewing angle or slight motion blur differs, choose none.",
            {"task": "Independent visual inspection of a proposed product take"},
            picture,
            model_override=model,
        )
        value = {
            "input": list(signature),
            **checked.model_dump(),
            "anatomy_review": {
                "initial": result.model_dump(),
                "second": checked.model_dump(),
            },
        }
    if fingerprint(video) != signature[0]:
        raise ValueError(f"Take changed during visual review: {video}")
    save(dest, value)
    return value
