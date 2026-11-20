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
        if zmin > 1.0:  # trellis, lantern, trim: not standing on a floor, not on a foundation plan
            continue
        faces = []
        for f in c["mesh"]["faces"]:
            pts = [verts[i] for i in f]
            if all(abs(v[2] - zmin) < 1e-4 for v in pts):
                faces.append([(v[0], v[1]) for v in pts])
        if faces:
            out.append({"id": p["id"], "component": c["name"], "kind": "column" if "Column" in c["name"] else ("glass" if "Glass" in c["name"] or c["name"].endswith("_Frame") else "wall"), "faces": faces})
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


# ---- elevations ---------------------------------------------------------------

LOOKING = {"+X": (1.0, 0.0), "-X": (-1.0, 0.0), "+Y": (0.0, 1.0), "-Y": (0.0, -1.0)}


def h_axis(looking: str) -> tuple[float, float]:
    """Plan direction that runs to the right in an elevation looking this way
    (the viewing direction crossed with up)."""
    lx, ly = LOOKING[looking]
    return ly, -lx


def load_elevation_sheet(calib_path: Path) -> np.ndarray:
    cal = yaml.safe_load(Path(calib_path).read_text(encoding="utf-8"))
    gray = cv2.imread(str(Path(calib_path).parent / cal["sheet"]), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(Path(calib_path).parent / cal["sheet"])
    turn = {None: None, "cw": cv2.ROTATE_90_CLOCKWISE, "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE, "180": cv2.ROTATE_180}[cal.get("rotate")]
    if turn is not None:
        gray = cv2.rotate(gray, turn)
    return flatten(gray)


def fit_elevations(calib_path: Path, points_m: dict[str, list[float]], levels_m: dict[str, float], datum_ft: float) -> dict[str, Any]:
    """Per view: pixel x = sx * h + bx (h in metres along the view's
    horizontal axis), pixel y = sy * z + by + tilt * x (z in metres above the
    datum)."""
    from architectworks.units import parse_length, to_metres

    cal = yaml.safe_load(Path(calib_path).read_text(encoding="utf-8"))
    views: dict[str, Any] = {}
    for name, v in cal["views"].items():
        hx, hy = h_axis(v["looking"])
        across = {k: x for k, x in v["across"].items() if k in points_m}
        if len(across) < 2:
            raise ValueError(f"{calib_path}: view {name} needs two named points the model knows")
        h = np.array([points_m[k][0] * hx + points_m[k][1] * hy for k in across])
        xs = np.array(list(across.values()), float)
        (sx, bx), *_ = np.linalg.lstsq(np.stack([h, np.ones_like(h)], 1), xs, rcond=None)
        res_x = {k: round(float(sx * hh + bx - x), 1) for k, hh, x in zip(across, h, xs)}
        rows, ys, labels = [], [], []
        for lv in v["levels"]:
            if "level" in lv:
                if lv["level"] not in levels_m:
                    continue
                z = levels_m[lv["level"]]
                labels.append(lv["level"])
            else:
                z = to_metres(parse_length(lv["elev"])) - datum_ft * FT
                labels.append(lv["elev"])
            px, py = lv["px"]
            rows.append([z, 1.0, px])
            ys.append(py)
        A = np.array(rows, float)
        if len({r[2] for r in rows}) == 1:
            A = A[:, :2]
        sol, *_ = np.linalg.lstsq(A, np.array(ys, float), rcond=None)
        sy, by, tilt = float(sol[0]), float(sol[1]), float(sol[2]) if len(sol) > 2 else 0.0
        res_y = {lab: round(float(sy * r[0] + by + tilt * r[2] - y), 1) for lab, r, y in zip(labels, rows, ys)}
        views[name] = {
            "looking": v["looking"],
            "h_axis": [hx, hy],
            "x": [float(sx), float(bx)],
            "y": [sy, by, tilt],
            "crop": v.get("crop"),
            "px_per_ft": [round(float(sx) * FT, 2), round(-sy * FT, 2)],
            "residual_px": {"across": res_x, "levels": res_y},
        }
    return {"sheet": cal["sheet"], "views": views}


def compare_elevation(view: dict[str, Any], render: dict[str, Any], render_dir: Path, drawing: np.ndarray, out_path: Path, title: str) -> dict[str, Any]:
    """The drawn elevation warped into the render's frame, above the render,
    above the two laid over each other (model outline in red)."""
    rgba = cv2.imread(str(render_dir / render["file"]), cv2.IMREAD_UNCHANGED)
    if rgba is None:
        raise FileNotFoundError(render_dir / render["file"])
    H_px, W_px = rgba.shape[:2]
    ppm, h0, ztop = render["px_per_m"], render["h0_m"], render["z_top_m"]
    sx, bx = view["x"]
    sy, by, tilt = view["y"]
    # render pixel (i, j) -> h = h0 + i/ppm, z = ztop - j/ppm -> drawing (x, y)
    x0 = sx * h0 + bx
    M = np.array([[sx / ppm, 0.0, x0], [tilt * sx / ppm, -sy / ppm, sy * ztop + by + tilt * x0]])
    warped = cv2.warpAffine(drawing, M, (W_px, H_px), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=255)
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0 if rgba.shape[2] == 4 else np.ones((H_px, W_px, 1), np.float32)
    on_white = (rgba[:, :, :3].astype(np.float32) * alpha + 255.0 * (1 - alpha)).astype(np.uint8)
    sheet_bgr = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
    mask = (alpha[:, :, 0] > 0.1).astype(np.uint8)  # glass is translucent but still the model
    edge = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    over = sheet_bgr.copy()
    tint = over.copy()
    tint[mask > 0] = (0.55 * tint[mask > 0] + 0.45 * np.array([120, 160, 255])).astype(np.uint8)
    over = tint
    over[edge] = (0, 0, 220)
    # how much of the drawing's ink falls inside the model's silhouette
    ink = warped < 150
    inside = float((ink & (mask > 0)).sum()) / max(1, int(ink.sum()))

    def band(img: np.ndarray, text: str) -> np.ndarray:
        bar = np.full((44, img.shape[1], 3), 255, np.uint8)
        cv2.putText(bar, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2, cv2.LINE_AA)
        return np.vstack([bar, img])

    sheet_img = band(sheet_bgr, f"{title}: the drawing, straightened to the model's scale")
    render_img = band(on_white, f"{title}: the model, rendered in the same frame")
    over_img = band(over, f"{title}: model outline (red) over the drawing")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), np.vstack([sheet_img, render_img, over_img]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return {"view": title, "image": str(out_path), "drawing_ink_inside_model": round(inside, 3)}
