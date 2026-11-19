"""Spec -> resolved model: components with meshes, placements with exact
transforms, and the cross-checks. Pure Python; no Blender."""

from __future__ import annotations

from typing import Any

from architectworks import __version__
from architectworks.geom import mesh as g
from architectworks.geom.xform import frame, rotation_z, matmul, translation
from architectworks.resolve.checks import run_spec_checks
from architectworks.resolve.model import CATEGORIES, Model, Placement
from architectworks.resolve.pavilion import EXPANDERS
from architectworks.spec.model import BuildingSpec

MARKER_MATERIAL = {"unreal": "M_AW_Marker", "color": "#e0b000", "translucent": False, "opacity": 1.0}
# Where the handedness marker stands, in Blender space (metres, degrees about Z).
MARKER_AT = (-6.0, 9.0, 0.0)
MARKER_YAW = 30.0


def marker_mesh() -> tuple[g.Mesh, list[g.Mesh]]:
    """An L on the ground with a post at its corner: 2.0 m along +X, 1.0 m along
    +Y, 1.5 m up. Every axis has a different extent and nothing is symmetric,
    so a swapped, mirrored or rotated import shows up in the bounds."""
    parts = [
        g.box(0.0, 2.0, 0.0, 0.2, 0.0, 0.2),
        g.box(0.0, 0.2, 0.2, 1.0, 0.0, 0.2),
        g.box(0.0, 0.2, 0.0, 0.2, 0.2, 1.5),
    ]
    return g.merge(parts), parts


def _hex_to_rgba(color: str, opacity: float) -> list[float]:
    c = color.lstrip("#")
    return [round(int(c[i : i + 2], 16) / 255.0, 4) for i in (0, 2, 4)] + [opacity]


def resolve(spec: BuildingSpec, *, marker: bool = True) -> dict[str, Any]:
    model = Model()
    model.checks.extend(run_spec_checks(spec))
    for el in spec.elements:
        EXPANDERS[el.pattern](spec, el, model)

    materials = {m.unreal: {"color": _hex_to_rgba(m.color, m.opacity), "translucent": m.translucent} for m in spec.types.materials.values()}
    if marker:
        mesh, parts = marker_mesh()
        name = model.unique(f"SM_{spec.building.code}_AW_HandednessMarker", "Fixtures", mesh, MARKER_MATERIAL["unreal"], parts, True, {"basis": "rule", "note": "pipeline test piece, not part of the building"})
        materials[MARKER_MATERIAL["unreal"]] = {"color": _hex_to_rgba(MARKER_MATERIAL["color"], 1.0), "translucent": False}
        model.place(
            Placement(
                id="aw.handedness_marker",
                component=name,
                category="Fixtures",
                matrix=matmul(translation(MARKER_AT), rotation_z(MARKER_YAW)),
                element="aw",
                basis="rule",
                confidence="high",
                source={"basis": "rule", "note": "asymmetric test piece for the Unreal import test"},
            )
        )

    order = {c: i for i, c in enumerate(CATEGORIES)}
    components = sorted(model.components.values(), key=lambda c: (order[c.category], c.name))
    return {
        "format": "architectworks.resolved",
        "version": 1,
        "generator": f"architectworks {__version__}",
        "building": spec.building.model_dump(),
        "space": {"units": "metres", "up": "+Z", "handedness": "right", "datum_level": spec.datum, "datum_elev": spec.levels[spec.datum].model_dump(mode="json")["elev"]},
        "materials": materials,
        "categories": list(CATEGORIES),
        "components": [c.to_json() for c in components],
        "placements": [p.to_json() for p in model.placements],
        "checks": [c.to_json() for c in model.checks],
        "derived": model.derived,
        "open_items": [o.model_dump(mode="json", exclude_none=True) for o in spec.open_items],
    }


__all__ = ["resolve", "marker_mesh", "MARKER_AT", "MARKER_YAW"]
