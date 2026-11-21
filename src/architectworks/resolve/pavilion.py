"""The radial pavilion pattern.

One spec entry ("regular hexagon, 8'-10" side, glazed on three sides, a stone
arc of 11'-8" radius from 238 to 303 degrees") expands into columns, headers,
doors, stone arcs, roof, rafters, compression ring and skylight, each at an
exact position, plus a footprint for the floor slab of its level.

Geometry rules, in the order they are applied:

* Plan: a regular n-gon about the control point; vertex k at angle
  rotation + k * 360/n. Glazed sides form one contiguous run (it may wrap
  past the last vertex).
* Glazed facets are planes at the stated pitch, rising from the eave line (the
  polygon side, at the height of the top of the rafters over the header).
* The compression ring: when its top is dimensioned, its radius is derived
  from the pitch (the hips reach the top of ring there); when only its radius
  is known, its top is derived instead. A radius scaled off a sheet is only a
  cross-check.
* The conical roof covers the angles the glass does not, out to the stone
  arc plus overhang where there is an arc and to the polygon edge elsewhere.
  Its slope runs from the top of ring down to the rafters bearing on the sill
  at the dimensioned top of masonry, and is checked against the general note.
  Every roof quad lies on one cone through its apex, so each is planar.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

from shapely.geometry import Polygon

from architectworks.geom import mesh as g
from architectworks.geom.mesh import Mesh, Vec3
from architectworks.geom.xform import Matrix, apply, frame, matmul, rotation_z, translation
from architectworks.resolve.model import Check, Footprint, Model, Placement, src_dict
from architectworks.spec.model import BuildingSpec, ComponentType, RadialPavilion, StoneArcSpec, contiguous_run
from architectworks.units import format_length, to_metres

Z: Vec3 = (0.0, 0.0, 1.0)
INCH = 0.0254


def camel(text: str) -> str:
    return "".join(part[:1].upper() + part[1:] for part in text.replace("-", "_").split("_") if part)


def fmt_m(metres: float) -> str:
    return format_length(Fraction(metres / INCH).limit_denominator(64))


def member_frame(start: Vec3, end: Vec3) -> tuple[Matrix, float]:
    """Frame for a beam whose top edge runs from ``start`` to ``end``: X along
    the beam, Y horizontal, Z perpendicular to X in the vertical plane."""
    d = g.sub(end, start)
    length = g.norm(d)
    x = g.unit(d)
    y = g.unit(g.cross(Z, x))
    z = g.cross(x, y)
    return frame(start, x, y, z), length


def ring_facet(lower0: Vec3, lower1: Vec3, r_ring: float, a0: float, a1: float, z_top: float, thickness: float, steps: int = 8) -> Mesh:
    """A roof facet from a straight lower edge up to an arc of the ring, the
    arc level at ``z_top`` all the way round, so the facet meets the round
    ring along a level line (a ruled surface, nearly flat)."""
    sections = []
    for i in range(steps + 1):
        f = i / steps
        lo = (lower0[0] + (lower1[0] - lower0[0]) * f, lower0[1] + (lower1[1] - lower0[1]) * f, lower0[2] + (lower1[2] - lower0[2]) * f)
        up = g.polar(r_ring, a0 + (a1 - a0) * f, z_top)
        sections.append([lo, up, (up[0], up[1], up[2] + thickness), (lo[0], lo[1], lo[2] + thickness)])
    return g.planar_faces(g.loft(sections))


def norm_span(start: float, end: float) -> tuple[float, float]:
    """(a0, a1) with 0 <= a0 < 360 and a0 < a1 <= a0 + 360."""
    a0 = start % 360.0
    span = (end - start) % 360.0 or 360.0
    return a0, a0 + span


def in_span(theta: float, a0: float, a1: float, eps: float = 1e-9) -> bool:
    return (theta - a0) % 360.0 <= (a1 - a0) + eps


@dataclass
class Ctx:
    """Shared values for one pavilion, in its local frame (vertex 0 on +X)."""

    spec: BuildingSpec
    el: RadialPavilion
    model: Model
    prefix: str
    n: int
    step: float
    half: float
    side: float
    R_c: float
    apothem: float
    P: Matrix
    z_f: float

    def vertex(self, k: int, r: float | None = None, z: float = 0.0) -> Vec3:
        return g.polar(self.R_c if r is None else r, k * self.step, z)

    def edge_radius(self, theta: float) -> float:
        k = math.floor(theta / self.step)
        return self.apothem / math.cos(math.radians(theta - (k + 0.5) * self.step))

    def level_z(self, level_id: str) -> float:
        return to_metres(self.spec.levels[level_id].elev - self.spec.levels[self.spec.datum].elev)

    def ctype(self, name: str) -> ComponentType:
        return self.spec.types.components[name]

    def material(self, t: ComponentType) -> tuple[str, bool]:
        m = self.spec.types.materials[t.material]
        return m.unreal, not m.translucent

    def tsrc(self, name: str) -> dict:
        return src_dict(self.spec.types.components[name].src, type=name)

    def place(self, pid: str, comp: str, local: Matrix, basis: str, confidence: str, source: dict) -> None:
        self.model.place(
            Placement(
                id=f"{self.el.id}.{pid}",
                component=comp,
                category=self.model.components[comp].category,
                matrix=matmul(self.P, local),
                element=self.el.id,
                basis=basis,
                confidence=confidence,
                source=source,
            )
        )

    def world_xy(self, p: Vec3) -> tuple[float, float]:
        w = apply(self.P, p)
        return (w[0], w[1])


def expand_pavilion(spec: BuildingSpec, el: RadialPavilion, model: Model) -> None:
    n = el.plan.sides
    step = 360.0 / n
    half = math.pi / n
    side = to_metres(el.plan.side)
    R_c = side / (2 * math.sin(half))
    ctrl = spec.control[el.center]
    z_f = to_metres(spec.levels[el.floor].elev - spec.levels[spec.datum].elev)
    P = matmul(translation((to_metres(ctrl.at[0]), to_metres(ctrl.at[1]), z_f)), rotation_z(el.plan.rotation))
    c = Ctx(spec, el, model, f"SM_{spec.building.code}", n, step, half, side, R_c, R_c * math.cos(half), P, z_f)

    g_start, g_count = contiguous_run(el.glazed_sides, n)
    G = [g_start + i for i in range(g_count)]  # unwrapped side indices
    glazed_vertices = list(range(g_start, g_start + g_count + 1)) if g_count else []
    rest = (float((g_start + g_count) * step), float(g_start * step + 360.0)) if g_count else (0.0, 360.0)
    arcs = [(a, *norm_span(a.start - el.plan.rotation, a.end - el.plan.rotation)) for a in el.stone_arcs]

    for k in range(n):
        model.points[f"{el.id}.v{k}"] = c.world_xy(c.vertex(k))
    model.points[el.id] = c.world_xy((0.0, 0.0, 0.0))

    col_outer = _columns_headers_doors(c, G, glazed_vertices)
    if el.trellis is not None and glazed_vertices:
        _trellis(c, G, glazed_vertices)
    tops = _stone_arcs(c, arcs, col_outer)
    _footprint(c, glazed_vertices, arcs)
    if el.roof is not None:
        _roof(c, G, glazed_vertices, rest, arcs, tops)


# ---- columns, headers, doors -----------------------------------------------------


def _columns_headers_doors(c: Ctx, G: list[int], glazed_vertices: list[int]) -> float:
    el, model = c.el, c.model
    col_t = c.ctype(el.columns.type)
    profile = [(to_metres(x), to_metres(y)) for x, y in c.spec.types.profiles[col_t.profile].points]
    col_outer = max(y for _, y in profile)
    if not glazed_vertices:
        return col_outer
    H_col = to_metres(col_t.height)
    col_mesh = g.prism(profile, 0.0, H_col)
    mat, nanite = c.material(col_t)
    col_name = model.shared(c.prefix, camel(el.columns.type), "Structure", col_mesh, mat, [col_mesh], nanite, c.tsrc(el.columns.type))
    plan_src = el.plan.src
    col_src = src_dict(el.columns.src or plan_src)
    for k in glazed_vertices:
        local = matmul(translation(c.vertex(k)), rotation_z(k * c.step - 90.0))
        c.place(f"column.v{k % c.n}", col_name, local, plan_src.basis, plan_src.confidence, col_src)

    hdr_t = c.ctype(el.header.type)
    w_h, d_h = (to_metres(v) for v in hdr_t.section)
    z_h = c.level_z(el.header.top) - c.z_f
    hdr_mesh = g.box(0.0, c.side, -w_h / 2, w_h / 2, -d_h, 0.0)
    mat, nanite = c.material(hdr_t)
    hdr_name = model.shared(c.prefix, camel(el.header.type), "Structure", hdr_mesh, mat, [hdr_mesh] if hdr_t.collision else [], nanite, c.tsrc(el.header.type))
    hdr_level = c.spec.levels[el.header.top].src
    for k in G:
        local, _ = member_frame(c.vertex(k, z=z_h), c.vertex(k + 1, z=z_h))
        c.place(f"header.s{k % c.n}", hdr_name, local, "dimensioned", hdr_level.confidence, src_dict(hdr_level, type=el.header.type))
    model.checks.append(close_check(f"{el.id}.header_depth", f"{el.id}: top of header minus floor minus column height equals the header depth", z_h - H_col, d_h, INCH / 8))

    door_t = c.ctype(el.doors.type)
    t_d = to_metres(door_t.thickness)
    # where the glazing line leaves the column, in the column's own frame
    e = g.unit(g.sub(c.vertex(1), c.vertex(0)))
    inset = g.ray_exit(profile, (-e[1], e[0]))
    L_door = c.side - 2 * inset
    mat, nanite = c.material(door_t)
    if el.doors.frame is not None:
        fr_t = c.ctype(el.doors.frame)
        fw, fd = (to_metres(v) for v in fr_t.section)
        frame_mesh, glass_mesh = g.glazing_unit(L_door, H_col, fw, fd, t_d, el.doors.panels)
        door_name = model.shared(c.prefix, camel(el.doors.type) + ("Pair" if el.doors.panels == 2 else "Pane"), "Envelope", glass_mesh, mat, [], nanite, c.tsrc(el.doors.type))
        f_mat, f_nan = c.material(fr_t)
        hull = g.box(0.0, L_door, -fd / 2, fd / 2, 0.0, H_col)
        frame_name = model.shared(c.prefix, camel(el.doors.type) + "Frame", "Envelope", frame_mesh, f_mat, [hull] if door_t.collision else [], f_nan, c.tsrc(el.doors.frame))
    else:
        door_mesh = g.box(0.0, L_door, -t_d / 2, t_d / 2, 0.0, H_col)
        door_name = model.shared(c.prefix, camel(el.doors.type), "Envelope", door_mesh, mat, [door_mesh] if door_t.collision else [], nanite, c.tsrc(el.doors.type))
        frame_name = None
    door_src = src_dict(el.doors.src) if el.doors.src else c.tsrc(el.doors.type)
    for k in G:
        p0, p1 = c.vertex(k), c.vertex(k + 1)
        d = g.unit(g.sub(p1, p0))
        m = frame(g.add(p0, g.scale(d, inset)), d, g.cross(Z, d), Z)
        c.place(f"door.s{k % c.n}", door_name, m, door_src.get("basis", "measured"), door_src.get("confidence", "low"), door_src)
        if frame_name is not None:
            c.place(f"door_frame.s{k % c.n}", frame_name, m, door_src.get("basis", "measured"), door_src.get("confidence", "low"), door_src)
    model.derived.setdefault(el.id, {}).update({"door_inset_from_column_cp": fmt_m(inset), "door_width": fmt_m(L_door)})
    return col_outer


# ---- stone arcs ----------------------------------------------------------------------


def segments_for(span: float, per_90: int) -> int:
    return max(2, math.ceil(span / 90.0 * per_90))


def _stone_arcs(c: Ctx, arcs, col_outer: float) -> list[float]:
    el, model = c.el, c.model
    tops = []
    for i, (a, a0, a1) in enumerate(arcs):
        wall_t = c.ctype(a.type)
        R_in, R_out, r_mid = wall_t.wall_radii(to_metres(a.radius_outside), to_metres)
        t_w = R_out - R_in
        z_m = c.level_z(a.top) - c.z_f
        tops.append(z_m)
        seg = segments_for(a1 - a0, a.segments_per_90)
        mesh = g.annular_sector(R_in, R_out, a0, a1, seg, 0.0, z_m)
        per = max(1, round(seg / max(1, round((a1 - a0) / 15.0))))
        hulls = [g.annular_sector(R_in, R_out, a0 + (a1 - a0) * j / seg, a0 + (a1 - a0) * min(j + per, seg) / seg, 1, 0.0, z_m) for j in range(0, seg, per)]
        mat, nanite = c.material(wall_t)
        suffix = "" if len(arcs) == 1 else str(i)
        name = model.unique(f"{c.prefix}_{camel(el.id)}_WallStoneArc{suffix}", "Envelope", mesh, mat, hulls if wall_t.collision else [], nanite, c.tsrc(a.type), uv="cylinder")
        ws = a.src or wall_t.src
        c.place(f"wall.arc{suffix}", name, rotation_z(0.0), ws.basis, ws.confidence, src_dict(ws, type=a.type))
        model.points[f"{el.id}.arc{i}.start"] = c.world_xy(g.polar(r_mid, a0))
        model.points[f"{el.id}.arc{i}.end"] = c.world_xy(g.polar(r_mid, a1))
        if a.returns:
            _returns(c, a, a0, a1, R_in, t_w, z_m, col_outer)
    return tops


def _returns(c: Ctx, a: StoneArcSpec, a0: float, a1: float, R_in: float, t_w: float, z_m: float, col_outer: float) -> None:
    model = c.model
    wall_t = c.ctype(a.type)
    mat, nanite = c.material(wall_t)
    assumed = {"basis": "assumed", "confidence": "low", "note": "straight returns from the end columns to the arc are assumed; the plan does not show them clearly"}
    for pid, ang, sgn in (("wall.return.start", a0, 1.0), ("wall.return.end", a1, -1.0)):
        on_vertex = abs(ang / c.step - round(ang / c.step)) < 1e-6
        r1 = (c.R_c + col_outer) if on_vertex else c.edge_radius(ang % 360.0)
        L_ret = R_in - r1
        if L_ret <= 0:
            continue
        ret_mesh = g.box(0.0, L_ret, -t_w / 2, t_w / 2, 0.0, z_m)
        ret_name = model.shared(c.prefix, camel(a.type) + "Return", "Envelope", ret_mesh, mat, [ret_mesh] if wall_t.collision else [], nanite, c.tsrc(a.type))
        u = g.polar(1.0, ang)
        tang = (-u[1], u[0], 0.0)
        origin = g.add(g.scale(u, r1), g.scale(tang, sgn * t_w / 2))
        c.place(pid, ret_name, frame(origin, u, tang, Z), "assumed", "low", assumed)


# ---- footprint ----------------------------------------------------------------------


def _footprint(c: Ctx, glazed_vertices: list[int], arcs) -> None:
    """The pavilion's share of its level's slab: glazed sides carried out by
    the edge offset, stone arcs to their outside face, open sides to the
    polygon edge."""
    el = c.el
    R_slab = c.R_c + to_metres(el.slab.edge_offset) / math.cos(c.half)
    if glazed_vertices:
        b0, b1 = glazed_vertices[-1] * c.step, glazed_vertices[0] * c.step + 360.0
    else:
        b0, b1 = 0.0, 360.0

    def radius(theta: float) -> float:
        for a, a0, a1 in arcs:
            if in_span(theta, a0, a1):
                return max(c.ctype(a.type).wall_radii(to_metres(a.radius_outside), to_metres)[1], c.edge_radius(theta % 360.0))
        return c.edge_radius(theta % 360.0)

    angles = set(g.angles(b0, b1, max(2, math.ceil((b1 - b0) / 2.0))))
    angles.update(k * c.step for k in range(-c.n, 3 * c.n) if b0 < k * c.step < b1)
    for _, a0, a1 in arcs:
        for edge in (a0, a1):
            for e in (edge, edge + 360.0, edge - 360.0):
                if b0 < e < b1:
                    angles.update({e - 1e-3, e + 1e-3})
    ring_pts = [(t, radius(t)) for t in sorted(angles)]
    if glazed_vertices:
        ring_pts = [(t, r) for i, (t, r) in enumerate(ring_pts) if not (i in (0, len(ring_pts) - 1) and r < R_slab)]
    pts = [c.vertex(k, r=R_slab)[:2] for k in glazed_vertices] + [g.polar(r, t)[:2] for t, r in ring_pts]
    local = Polygon(pts).buffer(0)
    world = Polygon([c.world_xy((x, y, 0.0)) for x, y in local.exterior.coords])

    # convex pieces for collision: a triangle per glazed side, sectors of at most 45 degrees elsewhere
    pieces: list[list[tuple[float, float]]] = []
    for i in range(len(glazed_vertices) - 1):
        pieces.append([(0.0, 0.0), c.vertex(glazed_vertices[i], r=R_slab)[:2], c.vertex(glazed_vertices[i + 1], r=R_slab)[:2]])
    count = max(1, math.ceil((b1 - b0) / 45.0))
    for i in range(count):
        s0, s1 = b0 + (b1 - b0) * i / count, b0 + (b1 - b0) * (i + 1) / count
        pieces.append([(0.0, 0.0)] + [g.polar(radius(t), t)[:2] for t in g.angles(s0, s1, 8)])
    hulls = [list(Polygon([c.world_xy((x, y, 0.0)) for x, y in p]).convex_hull.exterior.coords)[:-1] for p in pieces]
    c.model.footprints.append(
        Footprint(element=el.id, level=el.floor, slab_type=el.slab.type, polygon=world, hulls=hulls, source=src_dict(el.slab.src or c.ctype(el.slab.type).src, type=el.slab.type))
    )


# ---- roof ---------------------------------------------------------------------------


def _roof(c: Ctx, G: list[int], glazed_vertices: list[int], rest: tuple[float, float], arcs, tops: list[float]) -> None:
    el, model, roof = c.el, c.model, c.el.roof
    eid = el.id
    raf_t = c.ctype(roof.rafters.type)
    w_r, d_r = (to_metres(v) for v in raf_t.section)
    ring = roof.ring

    # ---- ring radius and top, and the glazed pitch ----
    pitch = z_e = tan_hip = None
    if roof.glazed is not None:
        pitch = math.radians(roof.glazed.pitch)
        z_h = c.level_z(el.header.top) - c.z_f
        z_e = z_h + d_r / math.cos(pitch)
        tan_hip = math.tan(pitch) * math.cos(c.half)
        if ring.top is not None:
            z_rt = c.level_z(ring.top) - c.z_f
            r_ring = (c.apothem - (z_rt - z_e) / math.tan(pitch)) / math.cos(c.half)
            ring_basis = "radius derived from pitch and top of ring"
            if ring.radius is not None:
                model.checks.append(close_check(f"{eid}.ring_radius_given", f"{eid}: ring radius given vs derived from the pitch", to_metres(ring.radius), r_ring, 3 * INCH))
                r_ring = to_metres(ring.radius)
        else:
            r_ring = to_metres(ring.radius)
            z_rt = z_e + (c.apothem - r_ring * math.cos(c.half)) * math.tan(pitch)
            ring_basis = "top of ring derived from pitch and ring radius"
        if not 0 < r_ring < c.apothem:
            raise ValueError(f"{eid}: ring radius {r_ring:.3f} m is not inside the plan; check pitch and levels")
    else:
        if ring.radius is None or ring.top is None:
            raise ValueError(f"{eid}: a roof without glazed facets needs the ring's radius and top")
        r_ring = to_metres(ring.radius)
        z_rt = c.level_z(ring.top) - c.z_f
        ring_basis = "radius and top as given"

    # ---- conical slope: from top of ring to rafters bearing on the first arc's sill ----
    sill_t = c.ctype(roof.sill.type)
    w_s, d_s = (to_metres(v) for v in sill_t.section)
    eave_overhang = to_metres(roof.conical.eave_overhang)
    if arcs:
        a = arcs[0][0]
        R_wc = c.ctype(a.type).wall_radii(to_metres(a.radius_outside), to_metres)[2]
        z_b = tops[0] + d_s
        slope = math.atan((z_rt - z_b - d_r) / (R_wc - r_ring))
        for _ in range(50):
            slope = math.atan((z_rt - z_b - d_r / math.cos(slope)) / (R_wc - r_ring))
        slope_basis = "derived from top of ring and top of masonry"
    elif tan_hip is not None:
        slope = math.atan(tan_hip)
        slope_basis = "continues the glazed hips"
    else:
        slope = math.radians(roof.conical.nominal_pitch or 30.0)
        slope_basis = "nominal pitch"
    tan_c = math.tan(slope)

    def cone_z(r: float) -> float:
        return z_rt - tan_c * (r - r_ring)

    def eave_radius(theta: float) -> float:
        for a, a0, a1 in arcs:
            if in_span(theta, a0, a1):
                return c.ctype(a.type).wall_radii(to_metres(a.radius_outside), to_metres)[1] + eave_overhang
        return c.edge_radius(theta % 360.0)

    # ---- glazed facets ----
    if roof.glazed is not None:
        rg_t = c.ctype(roof.glazed.panel)
        t_g = to_metres(rg_t.thickness)

        def facet_z(p: Vec3) -> float:
            u = g.polar(1.0, 0.5 * c.step)
            return z_e + (c.apothem - g.dot((p[0], p[1], 0.0), u)) * math.tan(pitch)

        v0, v1 = c.vertex(0, z=z_e), c.vertex(1, z=z_e)
        # the glass meets the steel ring along a level line at the top of the ring
        facet_mesh = ring_facet(v0, v1, r_ring, 0.0, c.step, z_rt, t_g)
        c0, c1 = g.polar(r_ring, 0.0, z_rt), g.polar(r_ring, c.step, z_rt)
        facet_hull = g.plate([v0, v1, c1, c0], t_g)
        mat, nanite = c.material(rg_t)
        facet_name = model.shared(c.prefix, camel(roof.glazed.panel), "Roof", facet_mesh, mat, [facet_hull] if rg_t.collision else [], nanite, c.tsrc(roof.glazed.panel))
        gl = roof.glazed.src
        for k in G:
            c.place(f"roof_glass.s{k % c.n}", facet_name, rotation_z(k * c.step), gl.basis, gl.confidence, src_dict(gl))
        # continuous blocking from the top of header up to the glass (D-1)
        if el.header.blocking is not None:
            blk_t = c.ctype(el.header.blocking)
            w_b = to_metres(blk_t.section[0])
            z_h = c.level_z(el.header.top) - c.z_f
            blk = g.box(0.0, c.side, -w_b / 2, w_b / 2, 0.0, z_e - z_h)
            mat, nanite = c.material(blk_t)
            blk_name = model.shared(c.prefix, camel(el.header.blocking), "Structure", blk, mat, [], nanite, c.tsrc(el.header.blocking))
            for k in G:
                p0, p1 = c.vertex(k, z=z_h), c.vertex(k + 1, z=z_h)
                d = g.unit(g.sub(p1, p0))
                c.place(f"blocking.s{k % c.n}", blk_name, frame(p0, d, g.cross(Z, d), Z), blk_t.src.basis, blk_t.src.confidence, c.tsrc(el.header.blocking))

    # ---- conical roof ----
    ranges = [rest] if roof.conical.span == "rest" else [(a0, a1) for _, a0, a1 in arcs]
    sh_t = c.ctype(roof.conical.type)
    t_v = to_metres(sh_t.thickness) / math.cos(slope)
    eps = 0.25

    def section(theta: float):
        R_e = eave_radius(theta)
        return g.radial_section(theta, [(r_ring, z_rt), (R_e, cone_z(R_e)), (R_e, cone_z(R_e) + t_v), (r_ring, z_rt + t_v)])

    cone_meshes, cone_hulls = [], []
    for b0, b1 in ranges:
        cuts = {b0, b1}
        for _, a0, a1 in arcs:
            for edge in (a0, a1):
                for e in (edge, edge + 360.0, edge - 360.0):
                    if b0 + eps < e < b1 - eps:
                        cuts.update({e - eps, e + eps})
        cuts.update(k * c.step for k in range(-c.n, 3 * c.n) if b0 < k * c.step < b1)
        cuts = sorted(cuts)
        samples = []
        for s0, s1 in zip(cuts, cuts[1:]):
            m = max(1, math.ceil((s1 - s0) / roof.conical.max_step - 1e-9))
            samples.extend(s0 + (s1 - s0) * j / m for j in range(m))
        samples.append(cuts[-1])
        sections = [section(t) for t in samples]
        cone_meshes.append(g.loft(sections))
        for i in range(0, len(sections) - 1, 2):
            cone_hulls.append(g.loft([sections[i], sections[min(i + 2, len(sections) - 1)]]))
    mat, nanite = c.material(sh_t)
    cone_name = model.unique(
        f"{c.prefix}_{camel(eid)}_RoofConical", "Roof", g.merge(cone_meshes), mat, [h for h in cone_hulls if g.is_convex(h, 1e-6)] if sh_t.collision else [], nanite, c.tsrc(roof.conical.type)
    , uv="cone")
    cs = roof.conical.src
    c.place("roof_conical", cone_name, rotation_z(0.0), cs.basis, cs.confidence, src_dict(cs, type=roof.conical.type, note=f"slope {slope_basis}"))

    # ---- eave trim: fascia over the rafter tails, frieze from wall to roof ----
    if roof.eave is not None:
        ev = roof.eave
        fas_t = c.ctype(ev.fascia)
        t_f, d_f = (to_metres(v) for v in fas_t.section)
        z_lo_drop = d_r / math.cos(slope)
        fas_meshes, fri_meshes = [], []
        for i, (a, a0, a1) in enumerate(arcs):
            seg = segments_for(a1 - a0, a.segments_per_90)
            R_e = c.ctype(a.type).wall_radii(to_metres(a.radius_outside), to_metres)[1] + eave_overhang
            z_e0 = cone_z(R_e)
            fas_meshes.append(g.annular_sector(R_e, R_e + t_f, a0, a1, seg, min(z_e0 - z_lo_drop, z_e0 + t_v - d_f), z_e0 + t_v))
            if ev.frieze is not None:
                R_in, R_out, _ = c.ctype(a.type).wall_radii(to_metres(a.radius_outside), to_metres)
                z_w = tops[i]
                if cone_z(R_out) > z_w + 1e-4:
                    sections = [g.radial_section(t, [(R_in, z_w), (R_out, z_w), (R_out, cone_z(R_out)), (R_in, cone_z(R_in))]) for t in g.angles(a0, a1, seg)]
                    fri_meshes.append(g.loft(sections))
        ev_src = src_dict(ev.src)
        if fas_meshes:
            mat, nanite = c.material(fas_t)
            name = model.unique(f"{c.prefix}_{camel(eid)}_Fascia", "Roof", g.merge(fas_meshes), mat, [], nanite, c.tsrc(ev.fascia), uv="cylinder")
            c.place("fascia", name, rotation_z(0.0), ev.src.basis, ev.src.confidence, ev_src)
        if fri_meshes:
            fri_t = c.ctype(ev.frieze)
            mat, nanite = c.material(fri_t)
            name = model.unique(f"{c.prefix}_{camel(eid)}_Frieze", "Envelope", g.merge(fri_meshes), mat, [], nanite, c.tsrc(ev.frieze), uv="cylinder")
            c.place("frieze", name, rotation_z(0.0), ev.src.basis, ev.src.confidence, ev_src)

    # ---- rafters ----
    mat, nanite = c.material(raf_t)
    raf_src = src_dict(roof.rafters.src) if roof.rafters.src else c.tsrc(roof.rafters.type)
    base = camel(roof.rafters.type)

    def rafter(pid: str, eave: Vec3, top: Vec3, basis: str) -> None:
        local, length = member_frame(eave, top)
        mesh = g.box(0.0, length, -w_r / 2, w_r / 2, -d_r, 0.0)
        name = model.shared(c.prefix, base, "Structure", mesh, mat, [mesh] if raf_t.collision else [], nanite, c.tsrc(roof.rafters.type))
        c.place(pid, name, local, basis, raf_src.get("confidence", "medium"), raf_src)

    if roof.glazed is not None:
        for k in glazed_vertices:
            rafter(f"rafter.hip.v{k % c.n}", c.vertex(k, z=z_e), c.vertex(k, r=r_ring, z=z_rt), "rule")
        if roof.rafters.glazed_midspan:
            for k in G:
                mid = (k + 0.5) * c.step
                t = g.polar(r_ring, mid)
                rafter(f"rafter.mid.s{k % c.n}", g.polar(c.apothem, mid, z_e), (t[0], t[1], z_rt), raf_src.get("basis", "rule"))
    idx = 0
    for b0, b1 in ranges:
        count = int(round((b1 - b0) / roof.rafters.cone_spacing))
        for i in range(1, count):
            ang = b0 + (b1 - b0) * i / count
            idx += 1
            R_e = eave_radius(ang)
            rafter(f"rafter.cone.{idx:02d}", g.polar(R_e, ang, cone_z(R_e)), g.polar(r_ring, ang, z_rt), raf_src.get("basis", "measured"))

    # ---- sills on the stone arcs ----
    for i, (a, a0, a1) in enumerate(arcs):
        R_wc = c.ctype(a.type).wall_radii(to_metres(a.radius_outside), to_metres)[2]
        sill_mesh = g.annular_sector(R_wc - w_s / 2, R_wc + w_s / 2, a0, a1, segments_for(a1 - a0, a.segments_per_90), tops[i], tops[i] + d_s)
        s_mat, s_nan = c.material(sill_t)
        suffix = "" if len(arcs) == 1 else str(i)
        sill_name = model.unique(f"{c.prefix}_{camel(eid)}_Sill{suffix}", "Structure", sill_mesh, s_mat, [], s_nan, c.tsrc(roof.sill.type), uv="cylinder")
        c.place(f"sill{suffix}", sill_name, rotation_z(0.0), sill_t.src.basis, sill_t.src.confidence, c.tsrc(roof.sill.type))

    # ---- compression ring and skylight ----
    if roof.lantern is not None:
        _lantern(c, r_ring, z_rt)

    ring_t = c.ctype(ring.type)
    w_ring, d_ring = (to_metres(v) for v in ring_t.section)
    mat, nanite = c.material(ring_t)
    # the band runs on up through the roofing as the skylight's curb
    z_band = z_rt + (t_v + to_metres(ring.curb) if ring.curb else 0.0)
    ring_name = model.unique(f"{c.prefix}_{camel(eid)}_CompressionRing", "Structure", g.ring(r_ring - w_ring, r_ring, z_rt - d_ring, z_band, 48), mat, [], nanite, c.tsrc(ring.type), uv="cylinder")
    ring_src = c.spec.levels[ring.top].src if ring.top else (ring.src or ring_t.src)
    c.place("ring", ring_name, rotation_z(0.0), "rule", ring_src.confidence, src_dict(ring_src, note=ring_basis))
    if roof.skylight is not None:
        dome_t = c.ctype(roof.skylight.type)
        mat, nanite = c.material(dome_t)
        dome_name = model.shared(c.prefix, camel(roof.skylight.type), "Roof", g.dome(r_ring - w_ring / 2, to_metres(dome_t.rise), segments=48, rings=8), mat, [], nanite, c.tsrc(roof.skylight.type))
        c.place("skylight", dome_name, translation((0.0, 0.0, z_band)), dome_t.src.basis, dome_t.src.confidence, c.tsrc(roof.skylight.type))

    # ---- derived values and cross-checks ----
    deg = math.degrees
    d = model.derived.setdefault(eid, {})
    d.update({"circumradius": fmt_m(c.R_c), "apothem": fmt_m(c.apothem), "ring_radius": fmt_m(r_ring), "ring_top_above_floor": fmt_m(z_rt), "conical_pitch_deg": round(deg(slope), 3)})
    if roof.glazed is not None:
        d.update({"eave_roof_surface_above_floor": fmt_m(z_e), "glazed_pitch_deg": round(roof.glazed.pitch, 3), "hip_pitch_deg": round(deg(math.atan(tan_hip)), 3)})
    if arcs:
        d["conical_eave_radius"] = fmt_m(eave_radius((arcs[0][1] + arcs[0][2]) / 2))
    if ring.measured_radius is not None:
        measured = to_metres(ring.measured_radius)
        diff = r_ring - measured
        model.checks.append(
            Check(
                id=f"{eid}.ring_radius",
                kind="derived",
                status="pass" if abs(diff) <= 3 * INCH else "conflict",
                message=f"{eid}: compression ring radius ({ring_basis}) is {fmt_m(r_ring)}; scaled off the sheet it is about {fmt_m(measured)} ({diff / INCH:+.1f} in). Geometry uses {fmt_m(r_ring)}.",
                values={"used": fmt_m(r_ring), "measured": fmt_m(measured), "difference_in": round(diff / INCH, 2)},
            )
        )
    if roof.conical.nominal_pitch is not None and arcs:
        diff_deg = deg(slope) - roof.conical.nominal_pitch
        model.checks.append(
            Check(
                id=f"{eid}.conical_pitch",
                kind="derived",
                status="pass" if abs(diff_deg) <= 1.0 else "conflict",
                message=f"{eid}: conical roof pitch {slope_basis} is {deg(slope):.1f} degrees; the general note says {roof.conical.nominal_pitch:g} ({diff_deg:+.1f}). Geometry follows the dimensioned levels.",
                values={"derived_deg": round(deg(slope), 3), "note_deg": roof.conical.nominal_pitch},
            )
        )
    if roof.glazed is not None and arcs and roof.conical.span == "rest":
        step_at_column = cone_z(c.R_c) - z_e
        model.checks.append(
            Check(
                id=f"{eid}.hip_step",
                kind="derived",
                status="pass" if abs(step_at_column) <= INCH else "conflict",
                message=f"{eid}: where the glass roof meets the conical roof, the two surfaces differ by {step_at_column / INCH:+.1f} in at the end columns (glazed hips at {deg(math.atan(tan_hip)):.1f} degrees, cone at {deg(slope):.1f})",
                values={"step_in": round(step_at_column / INCH, 2)},
            )
        )


def close_check(cid: str, what: str, measured: float, expected: float, tol: float) -> Check:
    diff = measured - expected
    return Check(
        id=cid,
        kind="derived",
        status="pass" if abs(diff) <= tol else "conflict",
        message=f"{what}: {fmt_m(measured)} vs {fmt_m(expected)} ({diff / INCH:+.2f} in)",
        values={"measured": fmt_m(measured), "expected": fmt_m(expected), "difference_in": round(diff / INCH, 3)},
    )


# ---- lantern -------------------------------------------------------------------------


def _lantern(c: Ctx, r_ring: float, z_rt: float) -> None:
    """Posts and fixed glass from the header to the lantern's header on the
    listed sides, and a glass facet from there to the compression ring over
    each of them."""
    el, model, lan = c.el, c.model, c.el.roof.lantern
    z_h0 = c.level_z(el.header.top) - c.z_f
    z_l = c.level_z(lan.top) - c.z_f
    hdr_t, post_t = c.ctype(lan.header), c.ctype(lan.post)
    w_h, d_h = (to_metres(v) for v in hdr_t.section)
    w_p = to_metres(post_t.section[0])
    if z_l - d_h <= z_h0:
        raise ValueError(f"{el.id}: the lantern's header is not above the pavilion's header")
    src = src_dict(lan.src)
    sides = sorted({k % c.n for k in lan.sides})
    verts = sorted({k for s_ in sides for k in (s_, (s_ + 1) % c.n)})
    post = g.box(-w_p / 2, w_p / 2, -w_p / 2, w_p / 2, z_h0, z_l - d_h)
    mat, nanite = c.material(post_t)
    post_name = model.shared(c.prefix, camel(lan.post) + "Post", "Structure", post, mat, [post], nanite, c.tsrc(lan.post))
    for k in verts:
        c.place(f"lantern.post.v{k}", post_name, matmul(translation(c.vertex(k)), rotation_z(k * c.step)), lan.src.basis, lan.src.confidence, src)
    hdr = g.box(0.0, c.side, -w_h / 2, w_h / 2, -d_h, 0.0)
    mat, nanite = c.material(hdr_t)
    hdr_name = model.shared(c.prefix, camel(lan.header), "Structure", hdr, mat, [hdr] if hdr_t.collision else [], nanite, c.tsrc(lan.header))
    pan_t = c.ctype(lan.panel)
    th = to_metres(pan_t.thickness)
    pane = g.box(w_p / 2, c.side - w_p / 2, -th / 2, th / 2, z_h0, z_l - d_h)
    mat, nanite = c.material(pan_t)
    pane_name = model.shared(c.prefix, camel(lan.panel) + "Lantern", "Envelope", pane, mat, [pane] if pan_t.collision else [], nanite, c.tsrc(lan.panel))
    skip = {k % c.n for k in lan.skip_headers}
    for k in sides:
        if k not in skip:
            local, _ = member_frame(c.vertex(k, z=z_l), c.vertex(k + 1, z=z_l))
            c.place(f"lantern.header.s{k}", hdr_name, local, lan.src.basis, lan.src.confidence, src)
        p0, p1 = c.vertex(k), c.vertex(k + 1)
        d = g.unit(g.sub(p1, p0))
        c.place(f"lantern.glass.s{k}", pane_name, frame(p0, d, g.cross(Z, d), Z), lan.src.basis, lan.src.confidence, src)
    # glass facets from the header's top edge to a level line round the ring
    rf_t = c.ctype(lan.roof_panel)
    t_g = to_metres(rf_t.thickness)
    tan_p = (z_rt - z_l) / (c.apothem - r_ring)
    facet = ring_facet(c.vertex(0, z=z_l), c.vertex(1, z=z_l), r_ring, 0.0, c.step, z_rt, t_g)
    mat, nanite = c.material(rf_t)
    facet_name = model.shared(c.prefix, camel(lan.roof_panel) + "Lantern", "Roof", facet, mat, [], nanite, c.tsrc(lan.roof_panel))
    for k in sides:
        c.place(f"lantern.roof.s{k}", facet_name, rotation_z(k * c.step), lan.src.basis, lan.src.confidence, src)
    model.derived.setdefault(el.id, {})["lantern_pitch_deg"] = round(math.degrees(math.atan(tan_p)), 1)


# ---- trellis -------------------------------------------------------------------------


def _trellis(c: Ctx, G: list[int], glazed_vertices: list[int]) -> None:
    """A pair of brackets along the rafter at each glazed column, level with
    the bottom of the header, and slats on top of them across each glazed
    side, outermost first, every ``spacing``."""
    el, model, tr = c.el, c.model, c.el.trellis
    hdr_t = c.ctype(el.header.type)
    z_hb = c.level_z(el.header.top) - c.z_f - to_metres(hdr_t.section[1])
    br_t, sl_t = c.ctype(tr.bracket), c.ctype(tr.slat)
    w_b, d_b = (to_metres(v) for v in br_t.section)
    w_s, d_s = (to_metres(v) for v in sl_t.section)
    w_r = to_metres(c.ctype(el.roof.rafters.type).section[0]) if el.roof is not None else 4 * INCH
    reach, spacing = to_metres(tr.reach), to_metres(tr.spacing)
    src = src_dict(tr.src)
    y0 = (w_r + w_b) / 2
    pair = g.merge([g.box(0.0, reach, y - w_b / 2, y + w_b / 2, z_hb, z_hb + d_b) for y in (-y0, y0)])
    mat, nanite = c.material(br_t)
    pair_name = model.shared(c.prefix, camel(tr.bracket) + "Pair", "Structure", pair, mat, [], nanite, c.tsrc(tr.bracket))
    for k in glazed_vertices:
        c.place(f"trellis.bracket.v{k % c.n}", pair_name, matmul(translation(c.vertex(k)), rotation_z(k * c.step)), tr.src.basis, tr.src.confidence, src)
    # slats for side 0, between the brackets at vertices 0 and 1
    slats = []
    s_ = reach - w_s / 2
    keep_clear = to_metres(hdr_t.section[0]) / 2 + 3 * INCH
    end_over = 3 * INCH
    while s_ * math.cos(c.half) > keep_clear:
        a = g.add(c.vertex(0), g.polar(s_, 0.0))
        b = g.add(c.vertex(1), g.polar(s_, c.step))
        d = g.unit(g.sub(b, a))
        length = g.norm(g.sub(b, a))
        local = frame(g.sub(a, g.scale(d, end_over)), d, g.cross(Z, d), Z)
        box = g.box(0.0, length + 2 * end_over, -w_s / 2, w_s / 2, z_hb + d_b, z_hb + d_b + d_s)
        slats.append(box.transformed(lambda v, m=local: apply(m, v)))
        s_ -= spacing
    if slats:
        mesh = g.merge(slats)
        mat, nanite = c.material(sl_t)
        sl_name = model.shared(c.prefix, camel(tr.slat) + "Slats", "Structure", mesh, mat, [], nanite, c.tsrc(tr.slat))
        for k in G:
            c.place(f"trellis.slats.s{k % c.n}", sl_name, rotation_z(k * c.step), tr.src.basis, tr.src.confidence, src)
    model.derived.setdefault(el.id, {})["trellis_slats_per_side"] = len(slats)
