"""The radial pavilion pattern.

One spec entry ("regular hexagon, 8'-10" side, glazed on three sides, stone
arc on the rest") expands into columns, headers, doors, walls, slab, roof,
rafters, compression ring and skylight, each at an exact position.

Geometry rules, in the order they are applied:

* Plan: a regular n-gon about the control point; vertex k at angle
  rotation + k * 360/n. Glazed sides form one contiguous run; the remaining
  span carries the stone arc wall.
* Glazed facets are planes at the stated pitch, rising from the eave line (the
  polygon side, at the height of the top of the rafters over the header).
* The compression ring radius is derived: it is where the hips reach the
  dimensioned top of ring. The radius scaled off the elevation is only a
  cross-check.
* The conical roof over the stone side runs from the top of ring down to the
  rafters bearing on the sill at the dimensioned top of masonry. Its slope is
  derived and checked against the general note.
"""

from __future__ import annotations

import math
from typing import Callable

from architectworks.geom import mesh as g
from architectworks.geom.mesh import Mesh, Vec3
from architectworks.geom.xform import Matrix, frame, matmul, rotation_z, translation
from architectworks.resolve.model import Check, Model, Placement, src_dict
from architectworks.spec.model import BuildingSpec, ComponentType, RadialPavilion
from architectworks.units import format_length, to_metres

Z: Vec3 = (0.0, 0.0, 1.0)
INCH = 0.0254


def camel(text: str) -> str:
    return "".join(part[:1].upper() + part[1:] for part in text.replace("-", "_").split("_") if part)


def fmt_m(metres: float) -> str:
    from fractions import Fraction

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


def expand_pavilion(spec: BuildingSpec, el: RadialPavilion, model: Model) -> None:
    code = spec.building.code
    prefix = f"SM_{code}"
    types = spec.types.components
    mats = spec.types.materials
    datum = spec.levels[spec.datum].elev

    def level_z(level_id: str) -> float:
        return to_metres(spec.levels[level_id].elev - datum)

    def ctype(name: str) -> ComponentType:
        return types[name]

    def material(t: ComponentType) -> tuple[str, bool]:
        m = mats[t.material]
        return m.unreal, not m.translucent

    def tsrc(name: str):
        return src_dict(types[name].src, type=name)

    # ---- plan ---------------------------------------------------------------
    n = el.plan.sides
    step = 360.0 / n
    half = math.pi / n
    side = to_metres(el.plan.side)
    R_c = side / (2 * math.sin(half))
    apothem = R_c * math.cos(half)
    ctrl = spec.control[el.center]
    cx, cy = to_metres(ctrl.at[0]), to_metres(ctrl.at[1])
    z_f = level_z(el.floor)
    P = matmul(translation((cx, cy, z_f)), rotation_z(el.plan.rotation))
    G = sorted(el.glazed_sides)
    g0, g1 = G[0], G[-1] + 1
    glazed_vertices = list(range(g0, g1 + 1))
    stone_a0, stone_a1 = g1 * step, g0 * step + 360.0
    eid = el.id

    def vertex(k: int, r: float = R_c, z: float = 0.0) -> Vec3:
        return g.polar(r, k * step, z)

    def place(pid: str, comp: str, local: Matrix, basis: str, confidence: str, source: dict) -> None:
        model.place(
            Placement(
                id=f"{eid}.{pid}",
                component=comp,
                category=model.components[comp].category,
                matrix=matmul(P, local),
                element=eid,
                basis=basis,
                confidence=confidence,
                source=source,
            )
        )

    plan_src = el.plan.src

    # ---- columns ------------------------------------------------------------
    col_t = ctype(el.columns.type)
    profile = [(to_metres(x), to_metres(y)) for x, y in spec.types.profiles[col_t.profile].points]
    H_col = to_metres(col_t.height)
    col_mesh = g.prism(profile, 0.0, H_col)
    mat, nanite = material(col_t)
    col_name = model.shared(prefix, camel(el.columns.type), "Structure", col_mesh, mat, [col_mesh], nanite, tsrc(el.columns.type))
    col_src = src_dict(el.columns.src or plan_src)
    for k in glazed_vertices:
        local = matmul(translation(vertex(k)), rotation_z(k * step - 90.0))
        place(f"column.v{k}", col_name, local, plan_src.basis, plan_src.confidence, col_src)
    col_outer = max(y for _, y in profile)

    # ---- headers ------------------------------------------------------------
    hdr_t = ctype(el.header.type)
    w_h, d_h = (to_metres(v) for v in hdr_t.section)
    z_h = level_z(el.header.top) - z_f
    hdr_mesh = g.box(0.0, side, -w_h / 2, w_h / 2, -d_h, 0.0)
    mat, nanite = material(hdr_t)
    hdr_name = model.shared(prefix, camel(el.header.type), "Structure", hdr_mesh, mat, [hdr_mesh] if hdr_t.collision else [], nanite, tsrc(el.header.type))
    for k in G:
        start, end = vertex(k, z=z_h), vertex(k + 1, z=z_h)
        local, _ = member_frame(start, end)
        place(f"header.s{k}", hdr_name, local, "dimensioned", spec.levels[el.header.top].src.confidence, src_dict(spec.levels[el.header.top].src, type=el.header.type))
    model.checks.append(
        _close_check(
            f"{eid}.header_depth",
            "top of header minus floor minus column height equals the header depth",
            measured=z_h - H_col,
            expected=d_h,
            tol=INCH / 8,
        )
    )

    # ---- doors --------------------------------------------------------------
    door_t = ctype(el.doors.type)
    t_d = to_metres(door_t.thickness)
    # where the glazing line leaves the column, in the column's own frame
    edge_dir_local = g.unit(g.sub(vertex(1), vertex(0)))
    a_rot = math.radians(-(0 * step - 90.0))
    in_col = (edge_dir_local[0] * math.cos(a_rot) - edge_dir_local[1] * math.sin(a_rot), edge_dir_local[0] * math.sin(a_rot) + edge_dir_local[1] * math.cos(a_rot))
    inset = g.ray_exit(profile, in_col)
    L_door = side - 2 * inset
    door_mesh = g.box(0.0, L_door, -t_d / 2, t_d / 2, 0.0, H_col)
    mat, nanite = material(door_t)
    door_name = model.shared(prefix, camel(el.doors.type), "Envelope", door_mesh, mat, [door_mesh] if door_t.collision else [], nanite, tsrc(el.doors.type))
    door_src = src_dict(el.doors.src) if el.doors.src else tsrc(el.doors.type)
    for k in G:
        a, b = vertex(k), vertex(k + 1)
        e = g.unit(g.sub(b, a))
        origin = g.add(a, g.scale(e, inset))
        place(f"door.s{k}", door_name, frame(origin, e, g.cross(Z, e), Z), door_src.get("basis", "measured"), door_src.get("confidence", "low"), door_src)

    # ---- roof numbers -------------------------------------------------------
    raf_t = ctype(el.roof.rafters.type)
    w_r, d_r = (to_metres(v) for v in raf_t.section)
    pitch = math.radians(el.roof.glazed.pitch)
    z_e = z_h + d_r / math.cos(pitch)
    z_rt = level_z(el.roof.ring.top) - z_f
    r_ring = (apothem - (z_rt - z_e) / math.tan(pitch)) / math.cos(half)
    if not 0 < r_ring < apothem:
        raise ValueError(f"{eid}: derived ring radius {r_ring:.3f} m is not inside the plan; check pitch and levels")
    tan_hip = math.tan(pitch) * math.cos(half)

    def facet_z(p: Vec3, k: int) -> float:
        u = g.polar(1.0, (k + 0.5) * step)
        return z_e + (apothem - g.dot((p[0], p[1], 0.0), u)) * math.tan(pitch)

    # stone side
    wall_t = ctype(el.stone_wall.type)
    R_out = to_metres(el.stone_wall.radius_outside)
    t_w = to_metres(wall_t.thickness)
    R_in, R_wc = R_out - t_w, R_out - t_w / 2
    z_m = level_z(el.stone_wall.top) - z_f
    sill_t = ctype(el.roof.sill.type)
    w_s, d_s = (to_metres(v) for v in sill_t.section)
    z_b = z_m + d_s
    slope = math.atan((z_rt - z_b - d_r) / (R_wc - r_ring))
    for _ in range(50):
        slope = math.atan((z_rt - z_b - d_r / math.cos(slope)) / (R_wc - r_ring))
    tan_c = math.tan(slope)

    def cone_z(r: float) -> float:
        return z_rt - tan_c * (r - r_ring)

    R_eave = R_out + to_metres(el.roof.conical.eave_overhang)

    # ---- glazed roof facets ---------------------------------------------------
    rg_t = ctype(el.roof.glazed.panel)
    t_g = to_metres(rg_t.thickness)
    arc_n = 8
    v0, v1 = vertex(0, z=z_e), vertex(1, z=z_e)
    inner = []
    for i in range(arc_n + 1):
        a = step * (arc_n - i) / arc_n
        p = g.polar(r_ring, a)
        inner.append((p[0], p[1], facet_z(p, 0)))
    facet_poly = [v0, v1] + inner
    facet_mesh = g.plate(facet_poly, t_g)
    facet_hull = g.plate([v0, v1, inner[0], inner[-1]], t_g)
    mat, nanite = material(rg_t)
    facet_name = model.shared(prefix, camel(el.roof.glazed.panel), "Roof", facet_mesh, mat, [facet_hull] if rg_t.collision else [], nanite, tsrc(el.roof.glazed.panel))
    gl_src = src_dict(el.roof.glazed.src)
    for k in G:
        place(f"roof_glass.s{k}", facet_name, rotation_z(k * step), el.roof.glazed.src.basis, el.roof.glazed.src.confidence, gl_src)

    # ---- rafters --------------------------------------------------------------
    mat, nanite = material(raf_t)
    raf_src = src_dict(el.roof.rafters.src) if el.roof.rafters.src else tsrc(el.roof.rafters.type)
    raf_basis = raf_src.get("basis", "rule")
    raf_conf = raf_src.get("confidence", "medium")
    base = camel(el.roof.rafters.type)

    def rafter(pid: str, eave: Vec3, top: Vec3, basis: str = raf_basis) -> None:
        local, length = member_frame(eave, top)
        mesh = g.box(0.0, length, -w_r / 2, w_r / 2, -d_r, 0.0)
        name = model.shared(prefix, base, "Structure", mesh, mat, [mesh] if raf_t.collision else [], nanite, tsrc(el.roof.rafters.type))
        place(pid, name, local, basis, raf_conf, raf_src)

    for k in glazed_vertices:
        rafter(f"rafter.hip.v{k}", vertex(k, z=z_e), vertex(k, r=r_ring, z=z_rt), "rule")
    if el.roof.rafters.glazed_midspan:
        for k in G:
            mid = (k + 0.5) * step
            e = g.polar(apothem, mid, z_e)
            t = g.polar(r_ring, mid)
            rafter(f"rafter.mid.s{k}", e, (t[0], t[1], z_e + (apothem - r_ring) * math.tan(pitch)))
    spacing = el.roof.rafters.cone_spacing
    count = int(round((stone_a1 - stone_a0) / spacing)) - 1
    for i in range(1, count + 1):
        a = stone_a0 + i * spacing
        rafter(f"rafter.cone.{i:02d}", g.polar(R_eave, a, cone_z(R_eave)), g.polar(r_ring, a, z_rt))

    # ---- stone wall, returns, sill -------------------------------------------
    seg_w = el.stone_wall.segments
    mat, nanite = material(wall_t)
    wall_mesh = g.annular_sector(R_in, R_out, stone_a0, stone_a1, seg_w, 0.0, z_m)
    per = max(1, seg_w // 12)
    wall_hulls = [
        g.annular_sector(R_in, R_out, stone_a0 + (stone_a1 - stone_a0) * i / seg_w, stone_a0 + (stone_a1 - stone_a0) * min(i + per, seg_w) / seg_w, 1, 0.0, z_m)
        for i in range(0, seg_w, per)
    ]
    wall_name = model.unique(f"{prefix}_{camel(eid)}_WallStoneArc", "Envelope", wall_mesh, mat, wall_hulls if wall_t.collision else [], nanite, tsrc(el.stone_wall.type))
    ws = el.stone_wall.src or wall_t.src
    place("wall.arc", wall_name, rotation_z(0.0), ws.basis, ws.confidence, src_dict(ws, type=el.stone_wall.type))

    if el.stone_wall.returns:
        r1 = R_c + col_outer
        L_ret = R_in - r1
        ret_mesh = g.box(0.0, L_ret, -t_w / 2, t_w / 2, 0.0, z_m)
        ret_name = model.shared(prefix, camel(el.stone_wall.type) + "Return", "Envelope", ret_mesh, mat, [ret_mesh] if wall_t.collision else [], nanite, tsrc(el.stone_wall.type))
        assumed = {"basis": "assumed", "confidence": "low", "note": "straight returns from the end columns to the arc are assumed; the plan does not show them clearly"}
        for pid, ang, sgn in (("wall.return.start", stone_a0, 1.0), ("wall.return.end", stone_a1, -1.0)):
            u = g.polar(1.0, ang)
            tang = (-u[1], u[0], 0.0)
            origin = g.add(g.scale(u, r1), g.scale(tang, sgn * t_w / 2))
            place(pid, ret_name, frame(origin, u, tang, Z), "assumed", "low", assumed)

    sill_mesh = g.annular_sector(R_wc - w_s / 2, R_wc + w_s / 2, stone_a0, stone_a1, seg_w, z_m, z_b)
    mat, nanite = material(sill_t)
    sill_name = model.unique(f"{prefix}_{camel(eid)}_Sill", "Structure", sill_mesh, mat, [], nanite, tsrc(el.roof.sill.type))
    place("sill", sill_name, rotation_z(0.0), sill_t.src.basis, sill_t.src.confidence, tsrc(el.roof.sill.type))

    # ---- conical roof ----------------------------------------------------------
    sh_t = ctype(el.roof.conical.type)
    t_v = to_metres(sh_t.thickness) / math.cos(slope)
    seg_c = el.roof.conical.segments

    def cone_section(a: float):
        return g.radial_section(a, [(r_ring, z_rt), (R_eave, cone_z(R_eave)), (R_eave, cone_z(R_eave) + t_v), (r_ring, z_rt + t_v)])

    cone_angles = g.angles(stone_a0, stone_a1, seg_c)
    cone_mesh = g.loft([cone_section(a) for a in cone_angles])
    hull_step = 2
    cone_hulls = [g.loft([cone_section(cone_angles[i]), cone_section(cone_angles[min(i + hull_step, seg_c)])]) for i in range(0, seg_c, hull_step)]
    mat, nanite = material(sh_t)
    cone_name = model.unique(f"{prefix}_{camel(eid)}_RoofConical", "Roof", cone_mesh, mat, cone_hulls if sh_t.collision else [], nanite, tsrc(el.roof.conical.type))
    cs = el.roof.conical.src
    place("roof_conical", cone_name, rotation_z(0.0), cs.basis, cs.confidence, src_dict(cs, type=el.roof.conical.type))

    # ---- compression ring and skylight --------------------------------------
    ring_t = ctype(el.roof.ring.type)
    w_ring, d_ring = (to_metres(v) for v in ring_t.section)
    ring_mesh = g.ring(r_ring - w_ring, r_ring, z_rt - d_ring, z_rt, 48)
    mat, nanite = material(ring_t)
    ring_name = model.unique(f"{prefix}_{camel(eid)}_CompressionRing", "Structure", ring_mesh, mat, [], nanite, tsrc(el.roof.ring.type))
    ring_level_src = spec.levels[el.roof.ring.top].src
    place("ring", ring_name, rotation_z(0.0), "rule", ring_level_src.confidence, src_dict(ring_level_src, note="radius derived from pitch and top of ring"))

    dome_t = ctype(el.roof.skylight.type)
    dome_mesh = g.dome(r_ring - w_ring / 2, to_metres(dome_t.rise), segments=48, rings=8)
    mat, nanite = material(dome_t)
    dome_name = model.shared(prefix, camel(el.roof.skylight.type), "Roof", dome_mesh, mat, [], nanite, tsrc(el.roof.skylight.type))
    place("skylight", dome_name, translation((0.0, 0.0, z_rt)), dome_t.src.basis, dome_t.src.confidence, tsrc(el.roof.skylight.type))

    # ---- slab -------------------------------------------------------------------
    slab_t = ctype(el.slab.type)
    t_slab = to_metres(slab_t.thickness)
    R_slab = R_c + to_metres(el.slab.edge_offset) / math.cos(half)
    glazed_pts = [vertex(k, r=R_slab)[:2] for k in glazed_vertices]
    arc_pts = [g.polar(R_out, a)[:2] for a in g.angles(stone_a0, stone_a1, seg_w)]
    slab_mesh = g.prism(glazed_pts + arc_pts, -t_slab, 0.0)
    slab_hulls = [g.prism([(0.0, 0.0), glazed_pts[i], glazed_pts[i + 1]], -t_slab, 0.0) for i in range(len(glazed_pts) - 1)]
    pieces = max(1, math.ceil((stone_a1 - stone_a0) / 90.0))
    for i in range(pieces):
        a0 = stone_a0 + (stone_a1 - stone_a0) * i / pieces
        a1 = stone_a0 + (stone_a1 - stone_a0) * (i + 1) / pieces
        pts = [g.polar(R_out, a)[:2] for a in g.angles(a0, a1, max(2, seg_w // pieces))]
        slab_hulls.append(g.prism([(0.0, 0.0)] + pts, -t_slab, 0.0))
    mat, nanite = material(slab_t)
    slab_name = model.unique(f"{prefix}_{camel(eid)}_Slab", "Ground", slab_mesh, mat, slab_hulls if slab_t.collision else [], nanite, tsrc(el.slab.type))
    ss = el.slab.src or slab_t.src
    place("slab", slab_name, rotation_z(0.0), ss.basis, ss.confidence, src_dict(ss, type=el.slab.type))

    # ---- derived values and cross-checks --------------------------------------
    deg = math.degrees
    model.derived[eid] = {
        "circumradius": fmt_m(R_c),
        "apothem": fmt_m(apothem),
        "eave_roof_surface_above_floor": fmt_m(z_e),
        "ring_radius": fmt_m(r_ring),
        "glazed_pitch_deg": round(el.roof.glazed.pitch, 3),
        "hip_pitch_deg": round(deg(math.atan(tan_hip)), 3),
        "conical_pitch_deg": round(deg(slope), 3),
        "conical_eave_radius": fmt_m(R_eave),
        "door_inset_from_column_cp": fmt_m(inset),
        "door_width": fmt_m(L_door),
    }
    if el.roof.ring.measured_radius is not None:
        measured = to_metres(el.roof.ring.measured_radius)
        diff = r_ring - measured
        model.checks.append(
            Check(
                id=f"{eid}.ring_radius",
                kind="derived",
                status="pass" if abs(diff) <= 3 * INCH else "conflict",
                message=(
                    f"compression ring radius derived from the {el.roof.glazed.pitch:g} degree pitch and top of ring is {fmt_m(r_ring)}; "
                    f"scaled off the elevation it is about {fmt_m(measured)} ({diff / INCH:+.1f} in). Geometry uses the derived value."
                ),
                values={"derived": fmt_m(r_ring), "measured": fmt_m(measured), "difference_in": round(diff / INCH, 2)},
            )
        )
    if el.roof.conical.nominal_pitch is not None:
        diff_deg = deg(slope) - el.roof.conical.nominal_pitch
        model.checks.append(
            Check(
                id=f"{eid}.conical_pitch",
                kind="derived",
                status="pass" if abs(diff_deg) <= 1.0 else "conflict",
                message=(
                    f"conical roof pitch derived from top of ring and top of masonry is {deg(slope):.1f} degrees; "
                    f"the general note says {el.roof.conical.nominal_pitch:g} ({diff_deg:+.1f}). Geometry follows the dimensioned levels."
                ),
                values={"derived_deg": round(deg(slope), 3), "note_deg": el.roof.conical.nominal_pitch},
            )
        )
    step_at_column = cone_z(R_c) - z_e
    model.checks.append(
        Check(
            id=f"{eid}.hip_step",
            kind="derived",
            status="pass" if abs(step_at_column) <= INCH else "conflict",
            message=(
                f"where the glass roof meets the conical roof, the two surfaces differ by {step_at_column / INCH:+.1f} in at the end columns "
                f"(glazed hips at {deg(math.atan(tan_hip)):.1f} degrees, cone at {deg(slope):.1f})"
            ),
            values={"step_in": round(step_at_column / INCH, 2)},
        )
    )


def _close_check(cid: str, what: str, measured: float, expected: float, tol: float) -> Check:
    diff = measured - expected
    return Check(
        id=cid,
        kind="derived",
        status="pass" if abs(diff) <= tol else "conflict",
        message=f"{what}: {fmt_m(measured)} vs {fmt_m(expected)} ({diff / INCH:+.2f} in)",
        values={"measured": fmt_m(measured), "expected": fmt_m(expected), "difference_in": round(diff / INCH, 3)},
    )


Expander = Callable[[BuildingSpec, RadialPavilion, Model], None]
EXPANDERS: dict[str, Expander] = {"radial_pavilion": expand_pavilion}
