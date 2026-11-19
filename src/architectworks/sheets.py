"""Sheets: fit a photographed plan to the model, straighten it, and check the
model against it (pipeline stages 1 and 7, for plans).

A calibration file lists pixel positions of named points: control points and
pavilion vertices whose plan positions the resolved model already knows. A
projective fit from plan to pixels then straightens the photo, and its
residuals say how well the spec agrees with the drawing. The overlay draws
the model's walls and columns onto the straightened sheet and scores how much
of their outline lands on ink.

Needs OpenCV (``pip install -e .[sheets]``).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    import cv2
except ImportError as err:  # pragma: no cover
    raise ImportError("sheet tools need OpenCV: pip install -e '.[sheets]'") from err

FT = 0.3048


def flatten(gray: np.ndarray, sigma: float = 25.0) -> np.ndarray:
    """Divide out the blotchy print background so linework stands out."""
    bg = cv2.GaussianBlur(gray, (0, 0), sigma)
    return cv2.divide(gray, bg, scale=255)


def fit(calib_path: Path, points_m: dict[str, list[float]]) -> dict[str, Any]:
    """Fit plan (metres) -> pixel homography from the calibration's named points."""
    cal = yaml.safe_load(Path(calib_path).read_text(encoding="utf-8"))
    names = [n for n in cal["points"] if n in points_m]
    missing = [n for n in cal["points"] if n not in points_m]
    if len(names) < 4:
        raise ValueError(f"{calib_path}: need at least 4 named points the model knows; have {names}")
    src = np.array([points_m[n] for n in names], np.float64)
    dst = np.array([cal["points"][n] for n in names], np.float64)
    H, _ = cv2.findHomography(src, dst, 0)
    back = cv2.perspectiveTransform(dst.reshape(-1, 1, 2), np.linalg.inv(H)).reshape(-1, 2)
    err_in = np.hypot(*(back - src).T) / 0.0254
    residuals = {n: round(float(e), 2) for n, e in zip(names, err_in)}
    probe = cv2.perspectiveTransform(np.array([[src.mean(axis=0)], [src.mean(axis=0) + [FT, 0]]], np.float64), H).reshape(-1, 2)
    return {
        "sheet": cal["sheet"],
        "H_plan_m_to_px": H.tolist(),
        "points_used": len(names),
        "points_unknown_to_model": missing,
        "residual_in": {"mean": round(float(err_in.mean()), 2), "max": round(float(err_in.max()), 2), "worst": max(residuals, key=residuals.get)},
        "residuals_in": residuals,
        "px_per_ft": round(float(np.hypot(*(probe[1] - probe[0]))), 2),
    }


def rectify(image_path: Path, H: np.ndarray, bounds_m: tuple[float, float, float, float], px_per_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Straighten the flattened sheet into plan space. Returns the image and
    the 3x3 matrix taking plan metres to its pixels."""
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(image_path)
    flat = flatten(gray)
    x0, y0, x1, y1 = bounds_m
    w, h = int(math.ceil((x1 - x0) * px_per_m)), int(math.ceil((y1 - y0) * px_per_m))
    out_to_plan = np.array([[1 / px_per_m, 0, x0], [0, -1 / px_per_m, y1], [0, 0, 1]])
    rect = cv2.warpPerspective(flat, H @ out_to_plan, (w, h), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP, borderValue=255)
    plan_to_rect = np.linalg.inv(out_to_plan)
    return rect, plan_to_rect


def _apply(m: list[list[float]], v: list[float]) -> tuple[float, float, float]:
    return tuple(m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2] + m[i][3] for i in range(3))  # type: ignore[return-value]


def plan_outlines(resolved: dict[str, Any], categories=("Envelope", "Structure")) -> list[dict[str, Any]]:
    """Plan outlines of what stands on the floor: the downward faces of walls,
    columns, glazing and masses, in world XY metres."""
    comps = {c["name"]: c for c in resolved["components"]}
    out = []
    for p in resolved["placements"]:
        c = comps[p["component"]]
        if p["category"] not in categories or "Rafter" in c["name"] or "Header" in c["name"] or "Ring" in c["name"] or "Sill" in c["name"]:
            continue
        verts = [_apply(p["matrix"], v) for v in c["mesh"]["verts"]]
        zmin = min(v[2] for v in verts)
        faces = []
        for f in c["mesh"]["faces"]:
            pts = [verts[i] for i in f]
            if all(abs(v[2] - zmin) < 1e-4 for v in pts):
                faces.append([(v[0], v[1]) for v in pts])
        if faces:
            out.append({"id": p["id"], "component": c["name"], "kind": "column" if "Column" in c["name"] else ("glass" if "Glass" in c["name"] else "wall"), "faces": faces})
    return out


def overlay(resolved: dict[str, Any], calib: dict[str, Any], image_path: Path, out_dir: Path, margin_m: float = 1.5, px_per_m: float = 80.0) -> dict[str, Any]:
    """Draw the model on the straightened sheet and score the agreement."""
    pts = np.array(list(resolved["points"].values()))
    outlines = plan_outlines(resolved)
    allxy = np.array([xy for o in outlines for f in o["faces"] for xy in f] + pts.tolist())
    x0, y0 = allxy.min(axis=0) - margin_m
    x1, y1 = allxy.max(axis=0) + margin_m
    H = np.array(calib["H_plan_m_to_px"])
    rect, P = rectify(image_path, H, (x0, y0, x1, y1), px_per_m)
    ink = (rect < 150).astype(np.uint8)
    near = cv2.dilate(ink, np.ones((7, 7), np.uint8))  # within ~4 cm at 80 px/m

    def to_px(xy):
        v = P @ np.array([xy[0], xy[1], 1.0])
        return (v[0] / v[2], v[1] / v[2])

    vis = cv2.cvtColor((rect * 0.55 + 115).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    scores = []
    colors = {"wall": (40, 40, 230), "column": (200, 60, 0), "glass": (0, 150, 0)}
    for o in outlines:
        hit = tot = 0
        for f in o["faces"]:
            poly = np.array([to_px(xy) for xy in f], np.float32)
            cv2.polylines(vis, [poly.astype(np.int32)], True, colors[o["kind"]], 2, cv2.LINE_AA)
            for i in range(len(poly)):
                a, b = poly[i], poly[(i + 1) % len(poly)]
                n = max(2, int(np.hypot(*(b - a)) / 4))
                for t in np.linspace(0, 1, n, endpoint=False):
                    u, v = (a + (b - a) * t).astype(int)
                    if 0 <= v < near.shape[0] and 0 <= u < near.shape[1]:
                        tot += 1
                        hit += int(near[v, u])
        scores.append({"id": o["id"], "kind": o["kind"], "on_ink": round(hit / tot, 3) if tot else None})
    for name, xy in resolved["points"].items():
        if "." not in name and not name.startswith("pav_") and xy:
            u, v = to_px(xy)
            cv2.drawMarker(vis, (int(u), int(v)), (0, 140, 255), cv2.MARKER_CROSS, 18, 2)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(calib["sheet"]).stem
    cv2.imwrite(str(out_dir / f"{stem}.rectified.png"), rect)
    cv2.imwrite(str(out_dir / f"{stem}.overlay.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
    by_kind: dict[str, list[float]] = {}
    for s in scores:
        if s["on_ink"] is not None:
            by_kind.setdefault(s["kind"], []).append(s["on_ink"])
    summary = {k: round(float(np.mean(v)), 3) for k, v in by_kind.items()}
    report = {
        "sheet": calib["sheet"],
        "fit": {k: calib[k] for k in ("points_used", "residual_in", "px_per_ft")},
        "residuals_in": calib["residuals_in"],
        "plan_bounds_m": [round(float(x0), 3), round(float(y0), 3), round(float(x1), 3), round(float(y1), 3)],
        "rectified_px_per_m": px_per_m,
        "on_ink_by_kind": summary,
        "elements": sorted(scores, key=lambda s: (s["on_ink"] is None, s["on_ink"] or 0)),
    }
    (out_dir / f"{stem}.check.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report
