"""Preserve actual product markings on the visible surface of a reconstructed mesh."""

import argparse
import math
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from scipy.optimize import minimize
import trimesh
from trimesh.visual.texture import TextureVisuals
from trimesh.visual.material import PBRMaterial
from .product_stage import Stage
from .core import save


def silhouette(rgb):
    bg = rgb[0, 0].astype("i2")
    return np.any(abs(rgb.astype("i2") - bg) > 2, axis=2).astype("uint8")


def match_view(asset, reference):
    stage = Stage(asset, (172, 252))
    ref = np.asarray(reference.resize((112, 168)))
    mask = (ref[:, :, 3] > 100).astype("uint8")
    target = cv2.cvtColor(ref[:, :, :3], cv2.COLOR_RGB2GRAY) * mask
    aspect = reference.width / reference.height

    def loss(params):
        angle, elev, roll = params
        rgb = np.asarray(stage.frame(angle, elev, roll, 0.90, (0, 0), 0))
        m = silhouette(rgb)
        x, y, w, h = cv2.boundingRect(m)
        if h < 5 or w < 5:
            return 2.0
        src = cv2.resize(
            cv2.cvtColor(rgb[y : y + h, x : x + w], cv2.COLOR_RGB2GRAY), (112, 168)
        )
        box = cv2.resize(m[y : y + h, x : x + w], (112, 168))
        common = (mask > 0) & (box > 0)
        if common.sum() < 100:
            return 2.0
        corr = np.corrcoef(src[common], target[common])[0, 1]
        return (
            1.0
            - float(corr)
            + 0.75 * np.mean(mask != box)
            + 0.22 * abs(math.log(w / h / aspect))
        )

    grid = [
        (a, e, r) for a in range(0, 81, 16) for e in (-14, 0, 14) for r in (-9, 0, 9)
    ]
    initial = min(grid, key=loss)
    opt = minimize(
        loss,
        initial,
        method="Nelder-Mead",
        options={
            "maxiter": 90,
            "initial_simplex": np.array(
                [
                    initial,
                    np.add(initial, [5, 0, 0]),
                    np.add(initial, [0, 5, 0]),
                    np.add(initial, [0, 0, 3]),
                ]
            ),
        },
    )
    return list(map(float, opt.x)), float(opt.fun)


def project(asset, photo, destination):
    reference = Image.open(photo).convert("RGBA")
    reference = reference.crop(
        reference.getchannel("A").point(lambda x: 255 if x > 40 else 0).getbbox()
    )
    angles, loss = match_view(asset, reference)
    size = (1100, 1650)
    stage = Stage(asset, size)
    original = np.asarray(stage.frame(*angles, 0.9, (0, 0), 0))
    mask = silhouette(original)
    x, y, w, h = cv2.boundingRect(mask)
    proj = np.zeros((size[1], size[0], 4), np.uint8)
    proj[y : y + h, x : x + w] = np.asarray(
        reference.resize((w, h), Image.Resampling.LANCZOS)
    )
    # Preserve manufacturer typography and the exact fan blade contours.
    aligned = proj
    rgb = aligned[:, :, :3].copy()
    holes = (aligned[:, :, 3] < 120).astype("uint8")
    rgb = cv2.inpaint(rgb, holes, 4, cv2.INPAINT_TELEA)
    image = Image.fromarray(rgb)
    depth = np.flipud(
        np.frombuffer(
            stage.fbo.read(attachment=-1, components=1, dtype="f4", alignment=1),
            dtype="f4",
        ).reshape(size[1], size[0])
    )
    meshes = list(trimesh.load_scene(asset, process=False).dump())
    verts = np.vstack([m.vertices for m in meshes])
    center = (verts.min(axis=0) + verts.max(axis=0)) / 2
    scale = np.ptp(verts, axis=0).max()
    output = trimesh.Scene()
    reference_material = PBRMaterial(
        name="CalibratedReference",
        baseColorTexture=image,
        metallicFactor=0.0,
        roughnessFactor=0.65,
    )
    selected = 0
    for mesh in meshes:
        p = (mesh.vertices - center) / scale
        clip = np.column_stack((p, np.ones(len(p)))) @ stage.mvp.T
        ndc = clip[:, :3] / clip[:, 3, None]
        uv = ndc[:, :2] / 2 + 0.5
        face = mesh.faces
        uv_mid = uv[face].mean(axis=1)
        xx = np.clip((uv_mid[:, 0] * size[0]).astype(int), 0, size[0] - 1)
        yy = np.clip(((1 - uv_mid[:, 1]) * size[1]).astype(int), 0, size[1] - 1)
        z = ndc[face, 2].mean(axis=1) * 0.5 + 0.5
        visible = ((z - depth[yy, xx]) < 0.0012) & (aligned[yy, xx, 3] > 128)
        selected += int(visible.sum())
        for front, valid in ((True, visible), (False, ~visible)):
            if not valid.any():
                continue
            part = mesh.copy()
            if front:
                part.visual = TextureVisuals(uv=uv.copy(), material=reference_material)
            part.update_faces(valid)
            part.remove_unreferenced_vertices()
            output.add_geometry(part)
    if not selected:
        raise ValueError("Reference projection missed every visible face")
    output.export(destination)
    save(
        destination.with_suffix(".json"),
        {
            "camera": angles,
            "registration_loss": round(loss, 4),
            "reference_faces": selected,
            "reference": photo.name,
        },
    )
    return angles


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ("asset", "photo", "output"):
        cli.add_argument(key, type=Path)
    args = cli.parse_args()
    print(project(args.asset, args.photo, args.output))
