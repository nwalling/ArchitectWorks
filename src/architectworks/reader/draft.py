"""Stage 2c: a plan reading becomes a draft spec.

Written dimensions beat pixels. The pixels' job is to say which written
dimension applies where:

1. Each dimension chain gives exact positions along its own axis for its
   ticks.
2. A center point or column takes a tick's value when an inked line runs from
   the tick to it, square to the chain within a small perspective tilt. When
   several ticks on one axis reach a feature, the one the fitted photo
   agrees with wins, and any disagreement is put up for review.
3. One chain per axis anchors that axis. Every other chain is placed by a
   projective fit of the photo, with its offset free, and then snapped to an
   exact link when the fit lands within 3 inches of one. Links are a feature
   both chains reach, or an inked extension line joining two ticks; they
   vote, center points and inked lines counting double because center
   lines are what a drafter sets a plan out from. Otherwise the chain's
   position stays measured, and says so.
4. Coordinates nothing dimensions are measured through the final fit, and
   every value carries a residual.
5. Pavilions come from their columns: sides and rotation from the column
   angles, side length from a typical note when there is one.

The draft is not a full Building Spec. It is the part a reading can supply,
in the spec's vocabulary, with a basis on every value, ready to be scored
against an answer key or merged into a spec by a reviewer.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from architectworks.reader.detect import Detections
from architectworks.reader.schema import PlanReading
from architectworks.units import LengthError, format_length, parse_length

SNAP_IN = 3.0


@dataclass
class Chain:
    index: int  # position in the reading
    side: str
    axis: str  # horizontal chains give X, vertical chains give Y
    ticks: np.ndarray  # (n, 2) pixels
    coords: list[Fraction | None]  # inches along the axis, relative to the chain's first tick
    offset: Fraction | None = None


@dataclass
class Feature:
    kind: str  # center | column
    label: str
    center: str  # for columns, the label of their center point
    px: np.ndarray
    links: dict[str, tuple[int, int]] = field(default_factory=dict)  # axis key -> (chain, tick) in use
    also: list[tuple[int, int]] = field(default_factory=list)  # other ticks that reach it
    hits: dict[str, list[tuple[float, int, int]]] = field(default_factory=dict)  # axis key -> (tilt, chain, tick)


def ink_between(flat: np.ndarray, a: np.ndarray, b: np.ndarray, threshold: int = 200) -> float:
    """Fraction of the straight path from ``a`` to ``b`` that runs over ink
    (within a pixel either side). Extension lines are faint and often
    dash-dotted: a real one scores 0.4 to 0.9 on these sheets."""
    h, w = flat.shape
    n = max(2, int(np.linalg.norm(b - a) / 1.5))
    t = np.linspace(0.0, 1.0, n)
    pts = a[None, :] + t[:, None] * (b - a)[None, :]
    d = (b - a) / (np.linalg.norm(b - a) or 1.0)
    nrm = np.array([-d[1], d[0]])
    best = np.full(n, 255.0)
    for off in (-1.0, 0.0, 1.0):
        q = pts + off * nrm[None, :]
        xi = np.clip(np.round(q[:, 0]).astype(int), 0, w - 1)
        yi = np.clip(np.round(q[:, 1]).astype(int), 0, h - 1)
        best = np.minimum(best, flat[yi, xi])
    return float((best < threshold).mean())


def _key(axis: str) -> str:
    return "x" if axis == "horizontal" else "y"


def _square_reach(a: np.ndarray, b: np.ndarray, along: np.ndarray, max_tilt_deg: float) -> float | None:
    """How far (degrees) the line from a to b leans off square to ``along``,
    or None past the allowance. Perspective leans real extension lines by up
    to a degree or two across a photographed sheet."""
    v = b - a
    dist = float(np.linalg.norm(v))
    if dist < 10:
        return None
    tilt = math.degrees(math.asin(min(1.0, abs(float(np.dot(v / dist, along))))))
    return tilt if tilt <= max_tilt_deg else None


class Draft:
    def __init__(self) -> None:
        self.control: dict[str, dict[str, Any]] = {}
        self.pavilions: list[dict[str, Any]] = []
        self.chains: list[dict[str, Any]] = []
        self.review: list[str] = []
        self.fit: dict[str, Any] = {}

    def to_json(self) -> dict[str, Any]:
        return {"format": "architectworks.draft", "version": 1, "fit": self.fit, "control": self.control, "pavilions": self.pavilions, "chains": self.chains, "review": self.review}


def build_draft(reading: PlanReading, flat: np.ndarray, det: Detections | None = None, max_tilt_deg: float = 2.0, min_ink: float = 0.4, pick_in: float = 2.0) -> Draft:
    draft = Draft()
    marks = det.by_id() if det is not None else {}

    def position(mark: str, x: float, y: float) -> np.ndarray:
        if mark and mark in marks:
            return np.array([marks[mark].x, marks[mark].y])
        return np.array([x, y], float)

    chains = _read_chains(reading, draft)
    feats = [Feature("center", cp.label, "", position(cp.mark, cp.x, cp.y)) for cp in reading.center_points]
    feats += [Feature("column", col.mark or f"column {j}", col.center, position(col.mark, col.x, col.y)) for j, col in enumerate(reading.columns)]

    # ---- which ticks could dimension which feature; start with the squarest ----
    for f in feats:
        found: dict[str, list[tuple[float, int, int]]] = defaultdict(list)
        for ci, ch in enumerate(chains):
            along = ch.ticks[-1] - ch.ticks[0]
            along = along / (np.linalg.norm(along) or 1.0)
            for k, t in enumerate(ch.ticks):
                if ch.coords[k] is None:
                    continue
                tilt = _square_reach(t, f.px, along, max_tilt_deg)
                if tilt is not None and ink_between(flat, t, f.px) >= min_ink:
                    found[_key(ch.axis)].append((tilt, ci, k))
        for key, hits in found.items():
            hits.sort()
            f.hits[key] = hits
            f.links[key] = (hits[0][1], hits[0][2])
            f.also.extend((ci, k) for _, ci, k in hits[1:])

    # ---- anchor one chain per axis; place the rest by the fit, then snap ----
    for axis in ("horizontal", "vertical"):
        idx = [i for i, c in enumerate(chains) if c.axis == axis]
        if idx:
            root = max(idx, key=lambda i: (sum(1 for f in feats if f.links.get(_key(axis), (None,))[0] == i), len(chains[i].ticks)))
            chains[root].offset = Fraction(0)
    fit, plan_xy = _fit(feats, chains)
    for _ in range(len(chains)):
        changed = False
        for ci, ch in enumerate(chains):
            if ch.offset is not None:
                continue
            cands = _link_candidates(ci, chains, feats, flat, max_tilt_deg, min_ink)
            fitted = fit["chain_offsets_in"][ci]
            groups: dict[Fraction, list] = {}
            for value, why, weight in cands:
                key = next((g for g in groups if abs(g - value) <= Fraction(1, 4)), value)
                groups.setdefault(key, []).append((why, weight))
            near = [(sum(w for _, w in v), -abs(float(g) - fitted), g, v) for g, v in groups.items() if abs(float(g) - fitted) <= SNAP_IN]
            if not near:
                continue
            votes, neg_gap, value, why = max(near, key=lambda t: (t[0], t[1]))
            ch.offset = value
            changed = True
            rivals = sorted(((sum(w for _, w in v), format_length(g)) for g, v in groups.items() if g != value and abs(float(g) - fitted) <= SNAP_IN), reverse=True)
            note = f"; rival readings {rivals}" if rivals else ""
            draft.review.append(f"chain {ci} ({ch.side}) placed by {why[0][0]} ({votes} votes){note}; the fit put it {-neg_gap:.1f} in away")
        if not changed:
            break
        fit, plan_xy = _fit(feats, chains)
    for ci, ch in enumerate(chains):
        if ch.offset is None:
            ch.offset = Fraction(round(fit["chain_offsets_in"][ci] * 4), 4)
            draft.review.append(f"chain {ci} ({ch.side}) shares no exact link with the anchored chains; its position {format_length(ch.offset)} is measured")
    # every chain is placed: each feature now takes the candidate the fitted photo agrees with
    fit, plan_xy = _fit(feats, chains)
    for fi, f in enumerate(feats):
        for key, hits in f.hits.items():
            ax = 0 if key == "x" else 1
            fitted_in = plan_xy[fi][ax] * 12
            scored = sorted((abs(float(chains[ci].offset + chains[ci].coords[k]) - fitted_in), ci, k) for _, ci, k in hits)
            gap, ci, k = scored[0]
            if gap <= pick_in:
                f.links[key] = (ci, k)
                f.also = [(c, kk) for _, c, kk in scored[1:]]
            else:
                f.links.pop(key, None)
                draft.review.append(f"{f.label}: no dimension within {pick_in:g} in of where the photo puts its {key} (nearest {gap:.1f} in); measured instead")
    fit, plan_xy = _fit(feats, chains)
    fit.pop("chain_offsets_in", None)
    draft.fit = fit

    # ---- control points ----
    for f, (X, Y) in zip(feats, plan_xy):
        if f.kind != "center":
            continue
        entry: dict[str, Any] = {}
        for key, val in (("x", X), ("y", Y)):
            if key in f.links:
                ci, k = f.links[key]
                entry[key] = format_length(chains[ci].offset + chains[ci].coords[k])
                entry[f"{key}_basis"] = f"dimensioned (chain {ci} {chains[ci].side}, tick {k})"
            else:
                entry[key] = format_length(Fraction(round(val * 12 * 4), 4))
                entry[f"{key}_basis"] = "measured"
        entry["px"] = [round(float(f.px[0]), 1), round(float(f.px[1]), 1)]
        draft.control[f.label] = entry
    for f in feats:
        for ci, k in f.also:
            ch = chains[ci]
            key = _key(ch.axis)
            if key in f.links:
                pc, pk = f.links[key]
                a, b = chains[pc].offset + chains[pc].coords[pk], ch.offset + ch.coords[k]
                if abs(a - b) > Fraction(1, 2):
                    draft.review.append(f"{f.label}: chain {pc} gives {key} = {format_length(a)}, chain {ci} gives {format_length(b)}; kept the nearer chain")

    _pavilions(reading, feats, plan_xy, draft)
    return draft


def _read_chains(reading: PlanReading, draft: Draft) -> list[Chain]:
    chains: list[Chain] = []
    for i, c in enumerate(reading.chains):
        parts: list[Fraction | None] = []
        for t in c.parts:
            try:
                parts.append(parse_length(t))
            except LengthError:
                parts.append(None)
                draft.review.append(f"chain {i} ({c.side}): could not read part {t!r}")
        ticks = np.array([[t.x, t.y] for t in c.ticks], float)
        if len(ticks) != len(parts) + 1:
            draft.review.append(f"chain {i} ({c.side}): {len(ticks)} ticks for {len(parts)} parts; skipped")
            continue
        sign = 1 if c.axis == "horizontal" else -1  # plan Y points up the sheet
        coords: list[Fraction | None] = [Fraction(0)]
        for part in parts:
            coords.append(None if part is None or coords[-1] is None else coords[-1] + sign * part)
        chains.append(Chain(i, c.side, c.axis, ticks, coords))
        _chain_report(draft, i, c, parts, ticks)
    return chains


def _link_candidates(ci: int, chains: list[Chain], feats: list[Feature], flat, max_tilt_deg: float, min_ink: float):
    """Exact offsets for chain ``ci`` implied by what it shares with chains
    already placed: a feature both reach, or an inked line joining two ticks."""
    ch = chains[ci]
    key = _key(ch.axis)
    out = []
    for f in feats:
        pairs = ([f.links[key]] if key in f.links else []) + f.also
        mine = [k for c, k in pairs if c == ci]
        theirs = [(c, k) for c, k in pairs if c != ci and chains[c].offset is not None and chains[c].axis == ch.axis]
        for k in mine:
            for c, ko in theirs:
                out.append((chains[c].offset + chains[c].coords[ko] - ch.coords[k], f"sharing {f.label} with chain {c}", 2 if f.kind == "center" else 1))
    for c, other in enumerate(chains):
        if c == ci or other.offset is None or other.axis != ch.axis:
            continue
        along = other.ticks[-1] - other.ticks[0]
        along = along / (np.linalg.norm(along) or 1.0)
        for ko, to in enumerate(other.ticks):
            if other.coords[ko] is None:
                continue
            for k, t in enumerate(ch.ticks):
                if ch.coords[k] is None:
                    continue
                if _square_reach(to, t, along, max_tilt_deg) is not None and ink_between(flat, to, t) >= min_ink:
                    out.append((other.offset + other.coords[ko] - ch.coords[k], f"an extension line from chain {c} tick {ko} to its tick {k}", 2))
    return out


def _pavilions(reading: PlanReading, feats: list[Feature], plan_xy, draft: Draft) -> None:
    side_notes: dict[str, Fraction] = {}
    radius_notes: dict[str, Fraction] = {}
    for n in reading.notes:
        try:
            val = parse_length(n.text.split(" ")[0]) if n.text else None
        except LengthError:
            val = None
        if val is None:
            continue
        for label in n.applies_to:
            if n.kind == "typical_side":
                side_notes[label] = val
            elif n.kind == "radius":
                radius_notes[label] = val
    centers = {f.label: i for i, f in enumerate(feats) if f.kind == "center"}
    by_center: dict[str, list[int]] = defaultdict(list)
    for i, f in enumerate(feats):
        if f.kind == "column":
            by_center[f.center].append(i)
    for label, cols in by_center.items():
        if label not in centers:
            draft.review.append(f"columns name an unknown center {label!r}")
            continue
        if len(cols) < 3:
            continue
        cx, cy = plan_xy[centers[label]]
        vec = [(plan_xy[i][0] - cx, plan_xy[i][1] - cy) for i in cols]
        angles = [math.degrees(math.atan2(dy, dx)) % 360 for dx, dy in vec]
        radii = [math.hypot(dx, dy) for dx, dy in vec]
        n_sides = 6
        rot_raw = math.degrees(math.atan2(sum(math.sin(math.radians(6 * a)) for a in angles), sum(math.cos(math.radians(6 * a)) for a in angles))) / 6 % 60
        snapped = round(rot_raw / 30) * 30 % 60
        rotation = snapped if abs(((rot_raw - snapped + 30) % 60) - 30) <= 3 else round(rot_raw, 1)
        spread = max(abs(((a - rotation + 30) % 60) - 30) for a in angles)
        vertices = sorted({round((a - rotation) / 60) % n_sides for a in angles})
        glazed = _contiguous([k for k in vertices if (k + 1) % n_sides in vertices], n_sides)
        mean_r_in = sum(radii) / len(radii) * 12
        if label in side_notes:
            side, side_basis = side_notes[label], "dimensioned (typical note)"
            if abs(mean_r_in - float(side)) > 3:
                draft.review.append(f"{label}: columns sit {mean_r_in - float(side):+.1f} in from the noted {format_length(side)} side")
        else:
            side, side_basis = Fraction(round(mean_r_in * 4), 4), "measured"
        entry = {
            "center": label,
            "sides": n_sides,
            "side": format_length(side),
            "side_basis": side_basis,
            "rotation_deg": rotation,
            "rotation_basis": "rule (snapped to 30 degrees)" if rotation == snapped else "measured",
            "columns_at_vertices": vertices,
            "glazed_sides": glazed,
            "column_angle_spread_deg": round(spread, 2),
        }
        if label in radius_notes:
            entry["stone_arc_radius"] = format_length(radius_notes[label])
        if spread > 4:
            draft.review.append(f"{label}: columns are up to {spread:.1f} degrees off a regular {n_sides}-gon")
        draft.pavilions.append(entry)


def _contiguous(sides: list[int], n: int) -> list[int]:
    if not sides:
        return []
    s = sorted(set(sides))
    for start in s:
        run = [(start + i) % n for i in range(len(s))]
        if sorted(run) == s:
            return run
    return s


def _chain_report(draft: Draft, i: int, c, parts, ticks: np.ndarray) -> None:
    """The scale-fit lie detector: each part's pixel length against the
    chain's median pixels per inch, and the parts against the overall."""
    d = np.linalg.norm(np.diff(ticks, axis=0), axis=1)
    ratios = [float(dk) / float(p) for dk, p in zip(d, parts) if p]
    med = float(np.median(ratios)) if ratios else float("nan")
    flags = []
    for k, (dk, p) in enumerate(zip(d, parts)):
        if p and med == med:
            expect = float(p) * med
            if abs(dk - expect) > max(3.0, 0.06 * expect):
                flags.append({"part": k, "lettered": c.parts[k], "pixels": round(float(dk), 1), "expected_pixels": round(expect, 1)})
    closes = None
    if c.overall:
        try:
            closes = all(parts) and sum(parts, Fraction(0)) == parse_length(c.overall)
        except LengthError:
            closes = None
    draft.chains.append({"index": i, "side": c.side, "axis": c.axis, "parts": c.parts, "overall": c.overall, "closes": closes, "px_per_inch": round(med, 4), "suspect_parts": flags})
    for f in flags:
        draft.review.append(f"chain {i} ({c.side}): part {f['part']} lettered {f['lettered']} spans {f['pixels']} px where the chain's scale expects {f['expected_pixels']}")
    if closes is False:
        draft.review.append(f"chain {i} ({c.side}): parts do not add up to the overall {c.overall}")


def _fit(feats: list[Feature], chains: list[Chain]):
    """Least squares over the plan->pixel homography (plan in feet), the
    offsets of chains not yet placed, each chain line's cross-axis position,
    and every feature coordinate no chain dimensions."""
    import cv2

    def known(f: Feature, key: str):
        if key not in f.links:
            return None
        ci, k = f.links[key]
        ch = chains[ci]
        return ("fixed", float(ch.offset + ch.coords[k]) / 12) if ch.offset is not None else ("chain", ci, float(ch.coords[k]) / 12)

    full = [f for f in feats if (known(f, "x") or ("",))[0] == "fixed" and (known(f, "y") or ("",))[0] == "fixed"]
    if len(full) < 4:
        raise ValueError(f"only {len(full)} features are dimensioned on both axes by the anchored chains; need 4 to fit the sheet")
    src = np.array([[known(f, "x")[1], known(f, "y")[1]] for f in full])
    dst = np.array([f.px for f in full])
    H0, _ = cv2.findHomography(src, dst, 0)
    H0 = H0 / H0[2, 2]
    Hinv = np.linalg.inv(H0)

    def to_plan(p):
        v = Hinv @ np.array([p[0], p[1], 1.0])
        return v[:2] / v[2]

    params = list(H0.ravel()[:8])
    slots: dict[Any, int] = {}

    def slot(name, guess):
        slots[name] = len(params)
        params.append(guess)

    for fi, f in enumerate(feats):
        g = to_plan(f.px)
        for ax, key in ((0, "x"), (1, "y")):
            if known(f, key) is None:
                slot(("feat", fi, ax), g[ax])
    for ci, ch in enumerate(chains):
        if ch.offset is None:
            g = to_plan(ch.ticks[0])
            slot(("off", ci), g[0] if ch.axis == "horizontal" else g[1])
        m = to_plan(ch.ticks.mean(axis=0))
        slot(("cross", ci), m[1] if ch.axis == "horizontal" else m[0])
    x0 = np.array(params, float)

    def coord(v, fi, f, ax, key):
        k = known(f, key)
        if k is None:
            return v[slots[("feat", fi, ax)]]
        if k[0] == "fixed":
            return k[1]
        return v[slots[("off", k[1])]] + k[2]

    def proj(H, X, Y):
        w = H @ np.array([X, Y, 1.0])
        return w[:2] / w[2]

    def residuals(v):
        H = np.append(v[:8], 1.0).reshape(3, 3)
        r = []
        for fi, f in enumerate(feats):
            r.extend(proj(H, coord(v, fi, f, 0, "x"), coord(v, fi, f, 1, "y")) - f.px)
        for ci, ch in enumerate(chains):
            off = float(ch.offset) / 12 if ch.offset is not None else v[slots[("off", ci)]]
            cross = v[slots[("cross", ci)]]
            for k, t in enumerate(ch.ticks):
                if ch.coords[k] is None:
                    continue
                a = off + float(ch.coords[k]) / 12
                r.extend(proj(H, *((a, cross) if ch.axis == "horizontal" else (cross, a))) - t)
        return np.array(r)

    sol = least_squares(residuals, x0, loss="soft_l1", f_scale=3.0)
    v = sol.x
    H = np.append(v[:8], 1.0).reshape(3, 3)
    plan_xy = [(coord(v, fi, f, 0, "x"), coord(v, fi, f, 1, "y")) for fi, f in enumerate(feats)]
    res = residuals(v).reshape(-1, 2)
    nf = len(feats)
    fe = np.hypot(res[:nf, 0], res[:nf, 1])
    te = np.hypot(res[nf:, 0], res[nf:, 1])
    cx, cy = np.mean(plan_xy, axis=0)
    ppf = float(np.linalg.norm(proj(H, cx + 1, cy) - proj(H, cx, cy)))
    worst = int(np.argmax(fe))
    return {
        "features": nf,
        "dimensioned_both_axes": len(full),
        "ticks": int(len(te)),
        "px_per_ft": round(ppf, 2),
        "feature_residual_in": {"mean": round(float(fe.mean()) / ppf * 12, 2), "max": round(float(fe.max()) / ppf * 12, 2), "worst": feats[worst].label},
        "tick_residual_in": {"mean": round(float(te.mean()) / ppf * 12, 2), "max": round(float(te.max()) / ppf * 12, 2)} if len(te) else None,
        "H_plan_ft_to_px": [[round(float(x), 9) for x in row] for row in H],
        "chain_offsets_in": {ci: float(v[slots[("off", ci)]]) * 12 for ci, ch in enumerate(chains) if ("off", ci) in slots},
    }, plan_xy
