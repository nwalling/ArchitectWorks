"""What the resolver produces: components (meshes), placements and checks."""

from __future__ import annotations

import hashlib
import json
import string
from dataclasses import dataclass, field
from typing import Any

from architectworks.geom.mesh import Mesh, bounds, box_uvs, is_closed, is_convex, volume
from architectworks.geom.xform import Matrix, rounded
from architectworks.spec.model import Source

CATEGORIES = ("Ground", "Structure", "Envelope", "Roof", "Fixtures")


def src_dict(src: Source | None, **extra: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if src is not None:
        out = {k: v for k, v in src.model_dump().items() if v is not None}
    out.update({k: v for k, v in extra.items() if v is not None})
    return out


@dataclass
class Component:
    name: str
    category: str
    mesh: Mesh
    material: str  # Unreal material name, e.g. M_Concrete
    collision: list[Mesh]
    nanite: bool
    source: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        bmin, bmax = bounds(self.mesh)
        return {
            "name": self.name,
            "category": self.category,
            "material_slots": [self.material],
            "nanite": self.nanite,
            "mesh": {**self.mesh.to_json(), "uv": [[[round(u, 6) + 0.0, round(v, 6) + 0.0] for u, v in f] for f in box_uvs(self.mesh)]},
            "collision": [c.to_json() for c in self.collision],
            "bounds": {"min": [round(c, 6) + 0.0 for c in bmin], "max": [round(c, 6) + 0.0 for c in bmax]},
            "volume_m3": round(volume(self.mesh), 6),
            "triangles": self.mesh.triangle_count,
            "source": self.source,
        }


@dataclass
class Placement:
    id: str
    component: str
    category: str
    matrix: Matrix
    element: str
    basis: str
    confidence: str
    source: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "component": self.component,
            "category": self.category,
            "element": self.element,
            "matrix": rounded(self.matrix),
            "basis": self.basis,
            "confidence": self.confidence,
            "source": self.source,
        }


@dataclass
class Footprint:
    """An element's share of its floor slab, in plan metres (world XY)."""

    element: str
    level: str
    slab_type: str
    polygon: Any  # shapely Polygon
    hulls: list[list[tuple[float, float]]]  # convex outlines for collision
    source: dict[str, Any]


@dataclass
class Check:
    id: str
    kind: str  # chain | rule | derived
    status: str  # pass | conflict | info
    message: str
    values: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "status": self.status, "message": self.message, "values": self.values}


def fingerprint(mesh: Mesh) -> str:
    payload = json.dumps(mesh.to_json(ndigits=5), sort_keys=True).encode()
    return hashlib.sha1(payload).hexdigest()


class Model:
    """Collects components (deduplicated by geometry) and placements."""

    def __init__(self) -> None:
        self.components: dict[str, Component] = {}
        self._variants: dict[str, list[tuple[str, str]]] = {}
        self.placements: list[Placement] = []
        self.checks: list[Check] = []
        self.derived: dict[str, dict[str, Any]] = {}
        self.points: dict[str, tuple[float, float]] = {}
        self.footprints: list[Footprint] = []

    def _validate(self, name: str, mesh: Mesh, collision: list[Mesh]) -> None:
        if not is_closed(mesh):
            raise ValueError(f"{name}: mesh is not closed")
        if volume(mesh) <= 0:
            raise ValueError(f"{name}: mesh has no volume")
        for i, hull in enumerate(collision):
            if not is_convex(hull):
                raise ValueError(f"{name}: collision piece {i} is not convex")

    def shared(self, prefix: str, base: str, category: str, mesh: Mesh, material: str, collision: list[Mesh], nanite: bool, source: dict) -> str:
        """Register geometry that many placements may share. Identical
        geometry under the same base name reuses one component; different
        geometry gets the next variant letter (_A, _B, ...)."""
        fp = fingerprint(mesh)
        variants = self._variants.setdefault(base, [])
        for name, existing in variants:
            if existing == fp:
                return name
        name = f"{prefix}_{base}_{string.ascii_uppercase[len(variants)]}"
        self._validate(name, mesh, collision)
        variants.append((name, fp))
        self.components[name] = Component(name, category, mesh, material, collision, nanite, source)
        return name

    def unique(self, name: str, category: str, mesh: Mesh, material: str, collision: list[Mesh], nanite: bool, source: dict) -> str:
        if name in self.components:
            raise ValueError(f"component {name} defined twice")
        self._validate(name, mesh, collision)
        self.components[name] = Component(name, category, mesh, material, collision, nanite, source)
        return name

    def place(self, placement: Placement) -> None:
        if placement.component not in self.components:
            raise KeyError(placement.component)
        if any(p.id == placement.id for p in self.placements):
            raise ValueError(f"placement id {placement.id} used twice")
        self.placements.append(placement)
