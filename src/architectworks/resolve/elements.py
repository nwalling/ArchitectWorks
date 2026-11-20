"""Elements that are not hexagons: stone rings, round masses, flues and links."""

from __future__ import annotations

import math

from shapely.geometry import Point, Polygon

from architectworks.geom import mesh as g
from architectworks.geom.xform import frame, matmul, rotation_z, translation
from architectworks.resolve.model import Check, Footprint, Model, Placement, src_dict
from architectworks.resolve.pavilion import INCH, Z, camel, fmt_m, member_frame, norm_span, segments_for
from architectworks.spec.model import BuildingSpec, Flue, Link, RoundMass, StoneRing
from architectworks.units import to_metres


def _z(spec: BuildingSpec, level: str) -> float:
    return to_metres(spec.levels[level].elev - spec.levels[spec.datum].elev)


def _material(spec: BuildingSpec, type_name: str) -> tuple[str, bool]:
    m = spec.types.materials[spec.types.components[type_name].material]
    return m.unreal, not m.translucent


def _center(spec: BuildingSpec, cp: str) -> tuple[float, float]:
    at = spec.control[cp].at
    return to_metres(at[0]), to_metres(at[1])


def expand_stone_ring(spec: BuildingSpec, el: StoneRing, model: Model) -> None:
    prefix = f"SM_{spec.building.code}"
    t = spec.types.components[el.type]
    cx, cy = _center(spec, el.center)
    z_f = _z(spec, el.floor)
    z_top = _z(spec, el.top) - z_f
    R_in, R_out, r_mid = t.wall_radii(to_metres(el.radius_outside), to_metres)
    t_w = R_out - R_in
    a0, a1 = norm_span(el.start, el.end)
    seg = segments_for(a1 - a0, el.segments_per_90)
    if a1 - a0 >= 360.0 - 1e-9:
        mesh = g.ring(R_out - t_w, R_out, 0.0, z_top, seg)
    else:
        mesh = g.annular_sector(R_out - t_w, R_out, a0, a1, seg, 0.0, z_top)
    pieces = max(1, round((a1 - a0) / 15.0))
    hulls = [g.annular_sector(R_out - t_w, R_out, a0 + (a1 - a0) * i / pieces, a0 + (a1 - a0) * (i + 1) / pieces, 1, 0.0, z_top) for i in range(pieces)]
    mat, nanite = _material(spec, el.type)
    name = model.unique(f"{prefix}_{camel(el.id)}_WallStone", "Envelope", mesh, mat, hulls if t.collision else [], nanite, src_dict(t.src, type=el.type))
    model.place(Placement(f"{el.id}.wall", name, "Envelope", translation((cx, cy, z_f)), el.id, el.src.basis, el.src.confidence, src_dict(el.src, type=el.type)))
    model.points[el.id] = (cx, cy)
    model.points[f"{el.id}.start"] = (cx + r_mid * math.cos(math.radians(a0)), cy + r_mid * math.sin(math.radians(a0)))
    model.points[f"{el.id}.end"] = (cx + r_mid * math.cos(math.radians(a1)), cy + r_mid * math.sin(math.radians(a1)))
    if el.slab is None or el.footprint == "none":
        return
    if el.footprint == "circle":
        poly = Point(cx, cy).buffer(R_out, quad_segs=24)
        hull_pts = [list(poly.exterior.coords)[:-1]]
    else:
        pts = [(cx, cy)] + [(cx + R_out * math.cos(math.radians(a)), cy + R_out * math.sin(math.radians(a))) for a in g.angles(a0, a1, seg)]
        poly = Polygon(pts).buffer(0)
        count = max(1, math.ceil((a1 - a0) / 45.0))
        hull_pts = []
        for i in range(count):
            s0, s1 = a0 + (a1 - a0) * i / count, a0 + (a1 - a0) * (i + 1) / count
            piece = [(cx, cy)] + [(cx + R_out * math.cos(math.radians(a)), cy + R_out * math.sin(math.radians(a))) for a in g.angles(s0, s1, 8)]
            hull_pts.append(list(Polygon(piece).convex_hull.exterior.coords)[:-1])
    model.footprints.append(Footprint(el.id, el.floor, el.slab.type, poly, hull_pts, src_dict(el.slab.src or el.src, type=el.slab.type)))


def expand_round_mass(spec: BuildingSpec, el: RoundMass, model: Model) -> None:
    prefix = f"SM_{spec.building.code}"
    t = spec.types.components[el.type]
    cx, cy = _center(spec, el.center)
    z_f = _z(spec, el.floor)
    r = to_metres(el.radius)
    poly = [(r * math.cos(2 * math.pi * i / 32), r * math.sin(2 * math.pi * i / 32)) for i in range(32)]
    mesh = g.prism(poly, 0.0, _z(spec, el.top) - z_f)
    mat, nanite = _material(spec, el.type)
    name = model.unique(f"{prefix}_{camel(el.id)}_RoundMass", "Envelope", mesh, mat, [mesh] if t.collision else [], nanite, src_dict(t.src, type=el.type))
    model.place(Placement(f"{el.id}.mass", name, "Envelope", translation((cx, cy, z_f)), el.id, el.src.basis, el.src.confidence, src_dict(el.src, type=el.type)))
    model.points[el.id] = (cx, cy)


def expand_flue(spec: BuildingSpec, el: Flue, model: Model) -> None:
    """A pipe from ``base`` up to a ball cap whose top is at ``top``."""
    prefix = f"SM_{spec.building.code}"
    t = spec.types.components[el.type]
    cx, cy = _center(spec, el.center)
    z_f = _z(spec, el.floor)
    z_base, z_top = _z(spec, el.base) - z_f, _z(spec, el.top) - z_f
    r_pipe, r_cap = to_metres(t.diameter) / 2, to_metres(t.cap_diameter) / 2
    z_cap = z_top - r_cap
    if z_cap - r_cap * 0.5 <= z_base:
        raise ValueError(f"{el.id}: the cap's top is too close to the base of the pipe")
    pipe = g.cylinder(r_pipe, z_base, z_cap, 24)
    cap = g.sphere(r_cap, 24, 12).transformed(lambda v: (v[0], v[1], v[2] + z_cap))
    mat, nanite = _material(spec, el.type)
    src = src_dict(el.src, type=el.type)
    name = model.unique(f"{prefix}_{camel(el.id)}", "Roof", g.merge([pipe, cap]), mat, [pipe] if t.collision else [], nanite, src)
    model.place(Placement(f"{el.id}.flue", name, "Roof", translation((cx, cy, z_f)), el.id, el.src.basis, el.src.confidence, src))
    model.points[el.id] = (cx, cy)


def expand_link(spec: BuildingSpec, el: Link, model: Model) -> None:
    prefix = f"SM_{spec.building.code}"
    for ref in (el.start, el.end):
        if ref not in model.points:
            raise ValueError(f"{el.id}: unknown point {ref!r}; known points include {sorted(model.points)[:12]}...")
    (x0, y0), (x1, y1) = model.points[el.start], model.points[el.end]
    length = math.hypot(x1 - x0, y1 - y0)
    if length < 1e-6:
        raise ValueError(f"{el.id}: start and end coincide")
    z_f = _z(spec, el.floor)
    d = ((x1 - x0) / length, (y1 - y0) / length, 0.0)
    y_axis = g.cross(Z, d)
    inset = min(to_metres(el.inset), length / 3)
    t = spec.types.components[el.type]
    mat, nanite = _material(spec, el.type)
    src = src_dict(el.src, type=el.type)
    if el.kind == "glazed":
        hdr = spec.types.components[el.header.type]
        w_h, d_h = (to_metres(v) for v in hdr.section)
        z_h = _z(spec, el.header.top) - z_f
        th = to_metres(t.thickness)
        pane = g.box(0.0, length - 2 * inset, -th / 2, th / 2, 0.0, z_h - d_h)
        pane_name = model.shared(prefix, camel(el.type), "Envelope", pane, mat, [pane] if t.collision else [], nanite, src_dict(t.src, type=el.type))
        origin = (x0 + d[0] * inset, y0 + d[1] * inset, z_f)
        model.place(Placement(f"{el.id}.pane", pane_name, "Envelope", frame(origin, d, y_axis, Z), el.id, el.src.basis, el.src.confidence, src))
        hdr_mesh = g.box(0.0, length, -w_h / 2, w_h / 2, -d_h, 0.0)
        h_mat, h_nan = _material(spec, el.header.type)
        hdr_name = model.shared(prefix, camel(el.header.type), "Structure", hdr_mesh, h_mat, [hdr_mesh] if hdr.collision else [], h_nan, src_dict(hdr.src, type=el.header.type))
        local, _ = member_frame((x0, y0, z_f + z_h), (x1, y1, z_f + z_h))
        model.place(Placement(f"{el.id}.header", hdr_name, "Structure", local, el.id, el.src.basis, el.src.confidence, src))
    else:
        th = to_metres(t.thickness)
        wall = g.box(0.0, length, -th / 2, th / 2, 0.0, _z(spec, el.top) - z_f)
        wall_name = model.shared(prefix, camel(el.type) + "Link", "Envelope", wall, mat, [wall] if t.collision else [], nanite, src_dict(t.src, type=el.type))
        model.place(Placement(f"{el.id}.wall", wall_name, "Envelope", frame((x0, y0, z_f), d, y_axis, Z), el.id, el.src.basis, el.src.confidence, src))
    if el.lettered is not None:
        want = to_metres(el.lettered)
        diff = length - want
        model.checks.append(
            Check(
                id=f"{el.id}.length",
                kind="derived",
                status="pass" if abs(diff) <= INCH / 2 else "conflict",
                message=f"{el.id}: {el.start} to {el.end} is {fmt_m(length)} from the geometry; the sheet letters {fmt_m(want)} ({diff / INCH:+.2f} in)",
                values={"geometry": fmt_m(length), "lettered": fmt_m(want), "difference_in": round(diff / INCH, 3)},
            )
        )
