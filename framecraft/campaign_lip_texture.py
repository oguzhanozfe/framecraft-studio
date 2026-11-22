"""Check whether a face edit preserves the presenter's lip colour and perioral skin.

Input patches are rectified against the original take. Measurement does not depend
on a mouth being open, or assume a particular lipstick or skin colour.
"""

import cv2
import numpy as np

_Y, _X = np.mgrid[-31.5:32:1, -55.5:56:1]
_LIP = (abs(_Y) < 13) & (abs(_X) < 30)
_CHEEK = (abs(_X) > 32) & (abs(_X) < 45) & (abs(_Y) < 13)
_CENTER = (abs(_X) < 15) & (abs(_Y) < 10)
_SIDE = (
    (abs(_X) > 17)
    & (abs(_X) < 45)
    & (abs(_Y) < 26)
    & ~((abs(_X) < 28) & (abs(_Y) < 12))
)


def _lip_profile(lab):
    skin = np.median(lab[_CHEEK], axis=0)
    pigment = lab[:, :, 1] - skin[1]
    dark = skin[0] - lab[:, :, 0]
    vermilion = _LIP & (pigment > 5) & (dark > 8)
    return {
        "coloured_area": int(vermilion.sum()),
        "coloured_width": int((vermilion.sum(0) >= 2).sum()),
        "bright_area": int((_CENTER & (dark < -16) & (pigment < 5)).sum()),
        "highlight": round(float(np.quantile(-dark[_CENTER], 0.8)), 2),
        "contour_darkness": round(float(np.quantile(dark[_CENTER], 0.75)), 2),
        "skin_lightness": float(skin[0]),
    }


def _upper_lip_detail(lightness):
    # The philtrum strip stays outside the true vermilion and oral cavity.
    # Black-hat morphology measures thin dark hairs; the Laplacian is a
    # separate sharpness indicator. Only a well-textured reference is eligible.
    lightness = np.ascontiguousarray(lightness, dtype=np.uint8)
    fine = cv2.morphologyEx(
        lightness,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    edges = np.abs(cv2.Laplacian(lightness, cv2.CV_32F))
    strip = (np.abs(_X) < 27) & (_Y < -7) & (_Y > -20)
    surrounding = (np.abs(_X) > 30) & (np.abs(_X) < 49) & (_Y < -9) & (_Y > -24)
    return (
        int(np.count_nonzero(fine[strip] > 8)),
        float(fine[strip].mean()),
        float(edges[strip].mean()),
        float(edges[surrounding].mean()),
    )


def _patch_metrics(original, rendered):
    a = cv2.cvtColor(original, cv2.COLOR_BGR2LAB).astype(np.float32)
    b = cv2.cvtColor(rendered, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref, edit = _lip_profile(a), _lip_profile(b)
    # Only compare pigment extent when the source visibly supports that measurement.
    # A closed mouth retains a dark, continuous rim. White teeth alone retain
    # wide coloured corners. A pale, disappearing lip loses both simultaneously.
    contraction = (
        ref["coloured_width"] >= 34
        and ref["coloured_area"] >= 240
        and edit["coloured_width"] < 0.63 * ref["coloured_width"]
        and edit["coloured_area"] < 0.5 * ref["coloured_area"]
        and edit["bright_area"] >= 153
        and edit["highlight"] - ref["highlight"] > 18
        and edit["contour_darkness"] < 30
    )
    # A foreign hair lock or cheek patch adds a coherent dark region on skin
    # beside the mouth. Open mouth cavities and the true lip seam are excluded.
    # The mouth-relative box can reach the hairline when the head turns. A dark
    # silhouette moving over background/shadow differs from new hair occluding
    # lit cheek. Count only pixels that were this subject's cheek colour first.
    source_skin = np.median(a[_CHEEK], axis=0)
    on_cheek = (
        np.linalg.norm(a[:, :, 1:] - source_skin[None, None, 1:], axis=2) < 15
    ) & (np.abs(a[:, :, 0] - source_skin[0]) < 36)
    extra = (
        _SIDE
        & on_cheek
        & (a[:, :, 0] - b[:, :, 0] > 40)
        & (b[:, :, 0] < edit["skin_lightness"] - 30)
    )
    comp = cv2.connectedComponentsWithStats(extra.astype(np.uint8), 8)[2][1:, 4]
    largest = int(comp.max(initial=0))
    detail_a, detail_b = _upper_lip_detail(a[:, :, 0]), _upper_lip_detail(b[:, :, 0])
    fibres, energy, sharpness, context = detail_a
    other, remaining, definition, context_b = detail_b
    context_ratio = context_b / max(1.0, context)
    lost_detail = (
        fibres >= 128
        and energy >= 5.0
        and sharpness >= 12
        and other < 0.49 * fibres
        and remaining < 0.55 * energy
        and definition / sharpness < min(0.55, 0.72 * context_ratio)
    )
    return {
        "source_fine_dark_pixels": fibres,
        "edited_fine_dark_pixels": other,
        "source_upper_lip_sharpness": round(sharpness, 3),
        "edited_upper_lip_sharpness": round(definition, 3),
        "upper_lip_texture_erased": bool(lost_detail),
        "source_lip_width": ref["coloured_width"],
        "edited_lip_width": edit["coloured_width"],
        "source_lip_area": ref["coloured_area"],
        "edited_lip_area": edit["coloured_area"],
        "bright_area": edit["bright_area"],
        "contour_darkness": edit["contour_darkness"],
        "new_dark_area": int(extra.sum()),
        "largest_dark_patch": largest,
        "pale_contraction": bool(contraction),
        "foreign_dark_patch": bool(largest >= 300 and extra.sum() >= 395),
    }


def _texture_metrics(original, rendered, face):
    span = float(np.linalg.norm(face[33] - face[263]))
    if span < 12 or original.shape[:2] != rendered.shape[:2]:
        raise ValueError(
            "Corresponding camera framing and a measurable face are required"
        )
    center = (face[61] + face[291]) / 2
    lip_axis = face[291] - face[61]
    tangent = lip_axis / max(1, np.linalg.norm(lip_axis))
    normal = np.asarray([-tangent[1], tangent[0]])
    pts = (
        center + 0.01 * span * (_X[:, :, None] * tangent + _Y[:, :, None] * normal)
    ).astype(np.float32)

    def crop(frame):
        return cv2.remap(
            frame,
            pts[:, :, 0],
            pts[:, :, 1],
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    return _patch_metrics(crop(original), crop(rendered))


def _texture_verdict(frames):
    """Report repeated visible defects; a single legitimate tooth highlight can be pale."""
    rows = [f["lip_texture"] for f in frames]
    defects = {}
    for kind in ("pale_contraction", "foreign_dark_patch", "upper_lip_texture_erased"):
        flagged = [i for i, row in enumerate(rows) if row.get(kind, False)]
        repeated = [
            i for i in flagged if any(i != j and abs(i - j) <= 2 for j in flagged)
        ]
        if repeated:
            defects[kind] = [frames[i]["time"] for i in repeated]
    return {
        "version": 3,
        "passed": not defects,
        "frames_checked": len(rows),
        "failure_times": sorted({t for times in defects.values() for t in times}),
        "defects": defects,
    }
