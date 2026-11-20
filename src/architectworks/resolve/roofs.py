"""Flat roofs between the pavilions, and the skylights that stand on them.

A flat roof's outline is traced on the roof plan from named points (pavilion
vertices, link ends, cores) and a few measured ones. Where another roof rises
above the flat roof, the flat roof is cut back to meet it, so the outline can
run generously into a pavilion and the cone or glass decides where the flat
roof stops. That is how two roofs meet on a building: along the line where
one surface passes through the level of the other.
"""

from __future__ import annotations

import math

import shapely
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import unary_union

from architectworks.geom import mesh as g
from architectworks.geom.xform import frame, translation
from architectworks.resolve.elements import _material, _z
from architectworks.resolve.floors import extrude
from architectworks.resolve.model import Model, Placement, src_dict
from architectworks.resolve.pavilion import Z, camel
from architectworks.spec.model import BuildingSpec, FlatRoof, FlatRoofSkylight
from architectworks.units import to_metres

MIN_PART_M2 = 0.05


def _point(model: Model, ref, where: str) -> tuple[float, float]:
    if isinstance(ref, str):
        if ref not in model.points:
            raise ValueError(f"{where}: unknown point {ref!r}")
        return model.points[ref]
    return to_metres(ref[0]), to_metres(ref[1])


def _clip_above(poly3: list[tuple[float, float, float]], z: float) -> list[tuple[float, float]]:
    """The part of a planar 3D polygon at or above height z, projected to plan."""
    out: list[tuple[float, float]] = []
    n = len(poly3)
    for i in range(n):
        a, b = poly3[i], poly3[(i + 1) % n]
        a_in, b_in = a[2] >= z, b[2] >= z
        if a_in:
            out.append((a[0], a[1]))
        if a_in != b_in:
            t = (z - a[2]) / (b[2] - a[2])
            out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
    return out


def roof_above(model: Model, z: float, skip_element: str) -> shapely.Geometry:
    """Plan region where any roof surface placed so far is at or above z."""
    pieces = []
    for p in model.placements:
        if p.category != "Roof" or p.element == skip_element:
            continue
        mesh = model.components[p.component].mesh
        m = p.matrix
        world = [(m[0][0] * x + m[0][1] * y + m[0][2] * zz + m[0][3], m[1][0] * x + m[1][1] * y + m[1][2] * zz + m[1][3], m[2][0] * x + m[2][1] * y + m[2][2] * zz + m[2][3]) for x, y, zz in mesh.verts]
        if max(v[2] for v in world) < z:
            continue
        for f in mesh.faces:
            pts = _clip_above([world[i] for i in f], z)
            if len(pts) >= 3:
                poly = Polygon(pts)
                if poly.is_valid and poly.area > 1e-9:
                    pieces.append(poly)
    return unary_union(pieces) if pieces else Polygon()


def expand_flat_roof(spec: BuildingSpec, el: FlatRoof, model: Model) -> None:
    prefix = f"SM_{spec.building.code}"
    t = spec.types.components[el.type]
    outline = [_point(model, r, f"{el.id}.outline") for r in el.outline]
    shape = Polygon(outline).buffer(0)
    for c in el.circles:
        shape = shape.union(Point(*_point(model, c.center, f"{el.id}.circles")).buffer(to_metres(c.radius), quad_segs=16))
    drawn = shape.area
    z_top = _z(spec, el.top)
    thick = to_metres(t.thickness)
    if el.trim:
        shape = shape.difference(roof_above(model, z_top, el.id))
    shape = shape.buffer(0)
    parts = list(shape.geoms) if isinstance(shape, MultiPolygon) else [shape]
    parts = [p.simplify(0.002) for p in parts if p.area > MIN_PART_M2]
    if not parts:
        raise ValueError(f"{el.id}: nothing is left of the flat roof after trimming")
    mesh = g.merge(extrude(p, z_top - thick, z_top) for p in parts)
    mat, nanite = _material(spec, el.type)
    src = src_dict(el.src, type=el.type)
    name = model.unique(f"{prefix}_{camel(el.id)}", "Roof", mesh, mat, [], nanite, src)
    model.place(Placement(f"{el.id}.roof", name, "Roof", translation((0.0, 0.0, 0.0)), el.id, el.src.basis, el.src.confidence, src))
    area = sum(p.area for p in parts)
    model.derived.setdefault("flat_roofs", {})[el.id] = {
        "area_m2": round(area, 3),
        "parts": len(parts),
        "trimmed_m2": round(drawn - area, 3),
        "top_m": round(z_top, 4),
    }
    for k, sk in enumerate(el.skylights):
        _skylight(spec, el, sk, k, z_top, model)


def ridge_beam(x0: float, x1: float, z: float, width: float, depth: float, over: float, lift: float, steps: int = 8) -> g.Mesh:
    """A beam along x from x0 - over to x1 + over, its underside at height z,
    straight between x0 and x1 and curving up by ``lift`` toward each tip."""
    xs = [x0 - over * (1 - i / steps) for i in range(steps)] + [x0 + (x1 - x0) * i / 4 for i in range(5)] + [x1 + over * (i + 1) / steps for i in range(steps)]

    def centre(x: float) -> tuple[float, float]:
        if over <= 0 or x0 <= x <= x1:
            return z + depth / 2, 0.0
        s = (x0 - x) / over if x < x0 else (x - x1) / over
        slope = 2 * lift * s / over * (-1 if x < x0 else 1)
        return z + depth / 2 + lift * s * s, slope

    sections = []
    for x in xs:
        zc, dz = centre(x)
        t = math.hypot(1.0, dz)
        nx, nz = -dz / t, 1.0 / t  # up, square to the beam
        sections.append([(x + a * nx * depth / 2, b * width / 2, zc + a * nz * depth / 2) for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
    return g.loft(sections)


def _skylight(spec: BuildingSpec, el: FlatRoof, sk: FlatRoofSkylight, k: int, z_roof: float, model: Model) -> None:
    """Built in a local frame: x along the ridge, y across it, z up from the
    roof; then turned and moved onto the roof."""
    prefix = f"SM_{spec.building.code}"
    L, W, R = to_metres(sk.length), to_metres(sk.width), to_metres(sk.rise)
    glass_t = spec.types.components[sk.type]
    frame_t = spec.types.components[sk.frame]
    th = to_metres(glass_t.thickness)
    fw, fd = (to_metres(v) for v in frame_t.section)
    over, lift = to_metres(sk.beam_overhang), to_metres(sk.beam_lift)
    if sk.form == "gable":
        h = W / 2
        slope = math.atan2(R, h)
        up_s, up_n = (0.0, -math.sin(slope), math.cos(slope)), (0.0, math.sin(slope), math.cos(slope))
        glass = g.merge([
            g.plate([(fw, -h, 0.0), (L - fw, -h, 0.0), (L - fw, 0.0, R), (fw, 0.0, R)], th, up=up_s),
            g.plate([(fw, 0.0, R), (L - fw, 0.0, R), (L - fw, h, 0.0), (fw, h, 0.0)], th, up=up_n),
        ])
        tri = [(-h, 0.0), (h, 0.0), (0.0, R)]
        ridge_y = 0.0
        parts = []
    else:
        slope = math.atan2(R, W)
        glass = g.plate([(fw, 0.0, 0.0), (L - fw, 0.0, 0.0), (L - fw, W, R), (fw, W, R)], th, up=(0.0, -math.sin(slope), math.cos(slope)))
        tri = [(0.0, 0.0), (W, 0.0), (W, R)]
        ridge_y = W
        parts = [g.box(fw, L - fw, W - fw, W, 0.0, R)]  # the back
    end = g.prism(tri, 0.0, fw).transformed(lambda v: (v[2], v[0], v[1]))
    parts += [end, end.transformed(lambda v: (v[0] + L - fw, v[1], v[2]))]
    parts.append(ridge_beam(0.0, L, R, fw, fd, over, lift).transformed(lambda v: (v[0], v[1] + ridge_y, v[2])))
    centre_x, centre_y = to_metres(sk.center[0]), to_metres(sk.center[1])
    a = math.radians(sk.axis)
    xdir = (math.cos(a), math.sin(a), 0.0)
    ydir = (-math.sin(a), math.cos(a), 0.0)
    off_y = 0.0 if sk.form == "gable" else W / 2
    origin = (centre_x - xdir[0] * L / 2 - ydir[0] * off_y, centre_y - xdir[1] * L / 2 - ydir[1] * off_y, z_roof)
    m = frame(origin, xdir, ydir, Z)
    src = src_dict(sk.src or el.src)
    gm, gn = _material(spec, sk.type)
    fm, fn = _material(spec, sk.frame)
    base = f"{prefix}_{camel(el.id)}_Skylight{k}"
    for part, mesh, mat, nan in (("glass", glass, gm, gn), ("frame", g.merge(parts), fm, fn)):
        name = model.unique(f"{base}{camel(part)}", "Roof", mesh, mat, [], nan, src)
        model.place(Placement(f"{el.id}.skylight{k}.{part}", name, "Roof", m, el.id, (sk.src or el.src).basis, (sk.src or el.src).confidence, src))
