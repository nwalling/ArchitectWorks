"""Gate 2: score a draft against an answer key (a hand-written spec).

The draft and the key may use different origins (each anchors its axes on a
different dimension line), so the draft is first shifted onto the key by the
translation most of its center points agree on. Then every center point and
every pavilion is compared, and every disagreement is listed.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from architectworks.spec.model import BuildingSpec, RadialPavilion
from architectworks.units import format_length, parse_length

EXACT_IN = 0.125
CLOSE_IN = 2.0
MATCH_IN = 18.0


def _xy(entry: dict[str, Any]) -> tuple[float, float]:
    return float(parse_length(entry["x"])), float(parse_length(entry["y"]))


def sheet_for_image(key: BuildingSpec, image: str | Path) -> str | None:
    """The key's sheet id for an image file, matched by file name."""
    name = Path(image).name
    return next((sid for sid, sh in key.sheets.items() if sh.file and Path(sh.file).name == name), None)


def score(draft: dict[str, Any], key: BuildingSpec, sheet: str | None = None) -> dict[str, Any]:
    """With ``sheet``, only the key's control points taken from that sheet
    are scored: a reading of one sheet cannot know points another sheet
    places."""
    key_pts = {cid: (float(cp.at[0]), float(cp.at[1])) for cid, cp in key.control.items() if sheet is None or cp.src.sheet == sheet}
    draft_pts = {label: _xy(e) for label, e in draft["control"].items()}

    # translation: the shift most draft points agree on (votes within 3 in)
    shifts = [(kx - dx, ky - dy) for dx, dy in draft_pts.values() for kx, ky in key_pts.values()]
    best, best_votes = (0.0, 0.0), -1
    for sx, sy in shifts:
        votes = [(tx, ty) for tx, ty in shifts if math.hypot(tx - sx, ty - sy) <= 3.0]
        if len(votes) > best_votes:
            best_votes = len(votes)
            best = (sum(t[0] for t in votes) / len(votes), sum(t[1] for t in votes) / len(votes))
    # snap the shift to the eighth inch that makes the most points exact
    sx, sy = best
    candidates = [(kx - dx, ky - dy) for dx, dy in draft_pts.values() for kx, ky in key_pts.values() if math.hypot(kx - dx - sx, ky - dy - sy) <= 3.0]

    def exact_count(t):
        return sum(1 for dx, dy in draft_pts.values() for kx, ky in key_pts.values() if math.hypot(dx + t[0] - kx, dy + t[1] - ky) <= EXACT_IN)

    if candidates:
        sx, sy = max(candidates, key=exact_count)

    points = []
    matched_keys: dict[str, str] = {}
    for label, (dx, dy) in draft_pts.items():
        X, Y = dx + sx, dy + sy
        kid, (kx, ky) = min(key_pts.items(), key=lambda kv: math.hypot(kv[1][0] - X, kv[1][1] - Y))
        err = math.hypot(kx - X, ky - Y)
        if err > MATCH_IN or kid in matched_keys.values():
            points.append({"draft": label, "key": None, "error_in": None})
            continue
        matched_keys[label] = kid
        basis = draft["control"][label]
        points.append(
            {
                "draft": label,
                "key": kid,
                "error_in": round(err, 3),
                "dx_in": round(X - kx, 3),
                "dy_in": round(Y - ky, 3),
                "draft_basis": [basis["x_basis"].split(" ")[0], basis["y_basis"].split(" ")[0]],
                "key_basis": key.control[kid].src.basis,
                "exact": err <= EXACT_IN,
                "close": err <= CLOSE_IN,
            }
        )
    missed = sorted(set(key_pts) - set(matched_keys.values()))

    # pavilions, matched through their centers
    key_pavs = {el.center: el for el in key.elements if isinstance(el, RadialPavilion)}
    pavs = []
    for p in draft["pavilions"]:
        kid = matched_keys.get(p["center"])
        el = key_pavs.get(kid)
        if el is None:
            pavs.append({"draft": p["center"], "key": None})
            continue
        rot_err = abs(((p["rotation_deg"] - el.plan.rotation + 30) % 60) - 30)
        checks = {
            "sides": p["sides"] == el.plan.sides,
            "side": parse_length(p["side"]) == el.plan.side,
            "rotation": rot_err <= 1.0,
            "glazed_sides": sorted(p["glazed_sides"]) == sorted(el.glazed_sides),
        }
        pavs.append({"draft": p["center"], "key": el.id, "checks": checks, "rotation_error_deg": round(rot_err, 2), "side": p["side"], "key_side": format_length(el.plan.side)})
    found_pavs = {p["key"] for p in pavs if p.get("key")}
    missed_pavs = sorted(el.id for el in key_pavs.values() if el.id not in found_pavs)

    n_items = len(points) + len(missed) + sum(len(p.get("checks", {})) or 1 for p in pavs) + len(missed_pavs)
    n_right = sum(1 for p in points if p.get("close")) + sum(sum(p["checks"].values()) for p in pavs if "checks" in p)
    exact = [p for p in points if p.get("exact")]
    return {
        "translation_in": [round(sx, 3), round(sy, 3)],
        "summary": {
            "items": n_items,
            "right": n_right,
            "score": round(n_right / n_items, 3) if n_items else None,
            "center_points": {"key": len(key_pts), "matched": len(matched_keys), "exact": len(exact), f"within_{CLOSE_IN:g}_in": sum(1 for p in points if p.get("close")), "missed": missed},
            "pavilions": {"key": len(key_pavs), "matched": sum(1 for p in pavs if p.get("key")), "all_checks_pass": sum(1 for p in pavs if p.get("checks") and all(p["checks"].values())), "missed": missed_pavs},
        },
        "points": points,
        "pavilions": pavs,
    }
