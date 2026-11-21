"""Build a textured relief of a readable catalog detail for a limited camera orbit."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image

from .audit import norm, read_text
from .campaign_edit import depth_session, fingerprint
from .core import save


def print_crop(shape, box, aspect=9 / 16):
    height, width = shape[:2]
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError("Invalid normalized text bounds")
    span = max(width * (x2 - x1) * 2.6, height * (y2 - y1) * aspect * 4)
    cw = min(width, round(height * aspect), max(2, round(span)))
    ch = min(height, max(2, round(cw / aspect)))
    x = max(0, min(width - cw, round(width * (x1 + x2) / 2 - cw / 2)))
    y = max(0, min(height - ch, round(height * (y1 + y2) / 2 - ch / 2)))
    if not (
        x <= width * x1
        and width * x2 <= x + cw
        and y <= height * y1
        and height * y2 <= y + ch
    ):
        raise ValueError("The selected mark does not fit a portrait macro")
    return x, y, x + cw, y + ch


def reconstruct(photo, words, model, asset):
    rgb = cv2.imread(str(photo))
    if rgb is None:
        raise ValueError(f"Unreadable catalog source: {photo}")
    found = [
        row
        for row in read_text(rgb)
        if row["confidence"] >= 0.75
        and norm(words) == norm(row["text"])
        and norm(words)
    ]
    if not found:
        raise ValueError(
            "The exact requested inscription must be visible in the source"
        )
    chosen = max(found, key=lambda row: row["confidence"])
    left, top, right, bottom = print_crop(rgb.shape, chosen["box"])
    image = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    network_input = cv2.resize(image, (392, 392)).astype(np.float32) / 255
    network_input -= np.array([0.485, 0.456, 0.406], np.float32)
    network_input /= np.array([0.229, 0.224, 0.225], np.float32)
    session = depth_session(model)
    inverse = session.run(
        None, {session.get_inputs()[0].name: network_input.transpose(2, 0, 1)[None]}
    )[0].squeeze()
    inverse = cv2.resize(inverse, (image.shape[1], image.shape[0]))
    crop = inverse[top:bottom, left:right]
    lo, hi = np.percentile(crop, [2, 98])
    if not np.isfinite(crop).all() or hi <= lo:
        raise ValueError("Depth model could not resolve this surface")
    texture = Image.fromarray(image[top:bottom, left:right])
    nx, ny = 161, 285
    z = np.clip((cv2.resize(crop, (nx, ny)) - lo) / (hi - lo), 0, 1) * 0.22
    u, v = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny))
    # Back-project to the reference camera. A second viewpoint reveals the surface relief.
    perspective = (1.9 - z) / 1.9
    vertices = np.stack(
        (
            (u - 0.5) * texture.width / texture.height * perspective,
            (0.5 - v) * perspective,
            z,
        ),
        axis=-1,
    ).reshape(-1, 3)
    grid = np.arange(nx * ny).reshape(ny, nx)
    a, b, c, d = grid[:-1, :-1], grid[:-1, 1:], grid[1:, :-1], grid[1:, 1:]
    faces = np.stack((a, c, b, b, c, d), axis=-1).reshape(-1, 3)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    material = trimesh.visual.material.PBRMaterial(
        name="CalibratedReference",
        baseColorTexture=texture,
        baseColorFactor=[255] * 4,
        metallicFactor=0,
        roughnessFactor=0.7,
    )
    mesh.visual = trimesh.visual.texture.TextureVisuals(
        uv=np.column_stack((u.ravel(), 1 - v.ravel())), material=material
    )
    asset.write_bytes(mesh.export(file_type="glb"))
    save(
        asset.with_suffix(".json"),
        {
            "kind": "visible_surface_relief",
            "source": str(photo),
            "source_sha256": fingerprint(photo),
            "model_sha256": fingerprint(model),
            "inscription": chosen,
            "pixel_crop": [left, top, right, bottom],
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("photo", type=Path)
    parser.add_argument("words")
    parser.add_argument("model", type=Path)
    parser.add_argument("asset", type=Path)
    args = parser.parse_args()
    reconstruct(args.photo, args.words, args.model, args.asset)
