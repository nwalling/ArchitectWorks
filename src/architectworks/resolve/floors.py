"""Floor slabs: one per level, from the footprints of every element on it.

Footprints on a level are unioned; gaps narrower than twice ``floors.gap_fill``
are closed (the corridors and halls between pavilions); then every other
level's footprints are cut away, so a sunken room keeps its own lower slab.
"""

from __future__ import annotations

from collections import defaultdict

import shapely
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from architectworks.geom import mesh as g
from architectworks.geom.xform import translation
from architectworks.resolve.model import Model, Placement
from architectworks.resolve.pavilion import camel
from architectworks.spec.model import BuildingSpec
from architectworks.units import to_metres


def extrude(poly: Polygon, z0: float, z1: float) -> g.Mesh:
    """A polygon with holes as a closed solid: constrained Delaunay caps,
    quads around every ring."""
    poly = shapely.set_precision(poly, 1e-6)
    tris = shapely.constrained_delaunay_triangles(poly)
    verts: list[g.Vec3] = []
    index: dict[tuple[float, float, float], int] = {}

    def vid(x: float, y: float, z: float) -> int:
        key = (round(x, 6), round(y, 6), z)
        if key not in index:
            index[key] = len(verts)
            verts.append((key[0], key[1], z))
        return index[key]

    faces: list[tuple[int, ...]] = []
    for tri in tris.geoms:
        (ax, ay), (bx, by), (cx, cy) = list(tri.exterior.coords)[:3]
        if (bx - ax) * (cy - ay) - (by - ay) * (cx - ax) < 0:
            (bx, by), (cx, cy) = (cx, cy), (bx, by)
        faces.append((vid(ax, ay, z1), vid(bx, by, z1), vid(cx, cy, z1)))
        faces.append((vid(ax, ay, z0), vid(cx, cy, z0), vid(bx, by, z0)))
    rings = [poly.exterior, *poly.interiors]
    for i_ring, ring in enumerate(rings):
        pts = list(ring.coords)[:-1]
        ccw = Polygon(pts).exterior.is_ccw
        outer = i_ring == 0  # exterior counter-clockwise, holes clockwise
        if ccw != outer:
            pts = pts[::-1]
        for i in range(len(pts)):
            (x0, y0), (x1, y1) = pts[i], pts[(i + 1) % len(pts)]
            faces.append((vid(x0, y0, z0), vid(x1, y1, z0), vid(x1, y1, z1), vid(x0, y0, z1)))
    return g.Mesh(verts, faces, [0] * len(faces)).oriented_outward()


def build_floors(spec: BuildingSpec, model: Model) -> None:
    if not model.footprints:
        return
    prefix = f"SM_{spec.building.code}"
    gap = to_metres(spec.floors.gap_fill)
    by_level: dict[tuple[str, str], list] = defaultdict(list)
    for fp in model.footprints:
        by_level[(fp.level, fp.slab_type)].append(fp)
    raw = {key: unary_union([fp.polygon for fp in fps]) for key, fps in by_level.items()}
    for (level, slab_type), fps in sorted(by_level.items()):
        shape = raw[(level, slab_type)]
        if gap > 0:
            shape = shape.buffer(gap, quad_segs=8, join_style="mitre").buffer(-gap, quad_segs=8, join_style="mitre").union(shape)
        others = [raw[k] for k in raw if k[0] != level]
        if others:
            shape = shape.difference(unary_union(others))
        shape = shape.buffer(0)
        parts = list(shape.geoms) if isinstance(shape, MultiPolygon) else [shape]
        parts = [p.simplify(0.002) for p in parts if p.area > 0.05]
        t = spec.types.components[slab_type]
        z_f = to_metres(spec.levels[level].elev - spec.levels[spec.datum].elev)
        thick = to_metres(t.thickness)
        mesh = g.merge(extrude(p, -thick, 0.0) for p in parts)
        hulls = []
        for fp in fps:
            for outline in fp.hulls:
                hull = g.prism(outline, -thick, 0.0)
                if g.is_convex(hull, 1e-6):
                    hulls.append(hull)
        m = spec.types.materials[t.material]
        name = model.unique(f"{prefix}_Slab{camel(level)}", "Ground", mesh, m.unreal, hulls if t.collision else [], not m.translucent, {"basis": "rule", "note": f"union of {len(fps)} footprints on {level}", "type": slab_type})
        basis = "assumed" if gap > 0 or any(fp.source.get("basis") == "assumed" for fp in fps) else "rule"
        model.place(
            Placement(
                id=f"slab.{level}",
                component=name,
                category="Ground",
                matrix=translation((0.0, 0.0, z_f)),
                element="floors",
                basis=basis,
                confidence="medium" if basis == "rule" else "low",
                source={"basis": basis, "note": f"footprints of {', '.join(fp.element for fp in fps)}" + (f"; gaps under {2 * gap:.2f} m closed" if gap > 0 else "")},
            )
        )
        model.derived.setdefault("floors", {})[level] = {"area_m2": round(sum(p.area for p in parts), 3), "parts": len(parts)}
