"""Recover the visible geometry and print of a round bottle from a clean front photograph."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image


def reconstruct(photo, target):
    image = Image.open(photo).convert("RGB")
    rgb = np.asarray(image)
    # Select the largest object against a near-white catalog background.
    fg = np.uint8(np.min(rgb, axis=2) < 238)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(fg)
    if count < 2:
        raise ValueError("No package silhouette in the source photograph")
    mask = labels == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    seen = np.nonzero(mask.any(axis=1))[0]
    height, width = mask.shape
    top, bottom = seen[0], seen[-1]
    if bottom - top < 40:
        raise ValueError("The front photograph is too small to reconstruct")
    rows = np.unique(
        np.linspace(top, bottom, min(300, bottom - top + 1)).round().astype(int)
    )
    phi = np.linspace(-np.pi, np.pi, 145)
    rings, uv = [], []
    for y in rows:
        hits = np.flatnonzero(mask[y])
        if not len(hits):
            raise ValueError("Incomplete bottle silhouette")
        left, right = hits[0], hits[-1]
        center, radius = (left + right) / 2, max(0.5, (right - left) / 2)
        rings.append(
            np.column_stack(
                (
                    radius * np.sin(phi),
                    np.full_like(phi, (top + bottom) / 2 - y),
                    radius * np.cos(phi),
                )
            )
        )
        # Map the forward half back onto its original photographed pixels. Only small
        # changes in viewing angle are supported by a single front reference.
        u = center + radius * np.sin(np.clip(phi, -np.pi / 2, np.pi / 2)) * 0.991
        uv.append(
            np.column_stack((u / (width - 1), np.full_like(phi, 1 - y / (height - 1))))
        )
    n = len(phi)
    faces = []
    for i in range(len(rows) - 1):
        for j in range(n - 1):
            a = i * n + j
            faces.extend(((a, a + n, a + 1), (a + 1, a + n, a + n + 1)))
    for j in range(1, n - 2):
        base = (len(rows) - 1) * n
        faces.extend(((0, j + 1, j), (base, base + j, base + j + 1)))
    material = trimesh.visual.material.PBRMaterial(
        name="CalibratedReference",
        baseColorTexture=image,
        baseColorFactor=[255, 255, 255, 255],
        metallicFactor=0,
        roughnessFactor=0.55,
    )
    surface = trimesh.Trimesh(vertices=np.vstack(rings), faces=faces, process=False)
    surface.visual = trimesh.visual.texture.TextureVisuals(
        uv=np.vstack(uv), material=material
    )
    target.write_bytes(surface.export(file_type="glb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("photo", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    reconstruct(args.photo, args.output)
