"""The Building Spec schema.

A spec is one YAML file per building, written the way the architect set the
building out: levels, control points and types first, placed elements second.
Every value that came off a sheet carries a ``src`` saying where it came from
and on what basis.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, PlainSerializer, model_validator

from architectworks.units import format_length, parse_angle, parse_length

Length = Annotated[
    Fraction,
    BeforeValidator(parse_length),
    PlainSerializer(lambda v: format_length(v), return_type=str),
]
Angle = Annotated[float, BeforeValidator(parse_angle), PlainSerializer(lambda v: f"{v:g}deg", return_type=str)]
Basis = Literal["dimensioned", "rule", "measured", "assumed"]
Confidence = Literal["high", "medium", "low"]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


class Source(Strict):
    """Where a value came from. ``basis`` orders authority:
    dimensioned > rule > measured > assumed."""

    sheet: str | None = None
    view: str | None = None
    basis: Basis
    confidence: Confidence | None = None
    quote: str | None = Field(None, description="Text as lettered on the sheet")
    note: str | None = None

    @model_validator(mode="after")
    def _default_confidence(self) -> "Source":
        if self.confidence is None:
            self.confidence = {"dimensioned": "high", "rule": "high", "measured": "low", "assumed": "low"}[self.basis]
        return self


class Building(Strict):
    id: str
    code: str = Field(pattern=r"^[A-Z][A-Za-z0-9]{0,11}$", description="Short code used in asset names")
    name: str


class Sheet(Strict):
    title: str
    file: str | None = None
    scale: str | None = None
    calibration: str | None = Field(None, description="pixel positions of named points, for fitting the sheet to the plan")
    note: str | None = None


class Level(Strict):
    elev: Length
    src: Source


class ControlPoint(Strict):
    at: tuple[Length, Length]
    src: Source


class Material(Strict):
    unreal: str = Field(pattern=r"^M_[A-Za-z0-9_]+$")
    color: str = Field(pattern=r"^#[0-9a-fA-F]{6}$")
    translucent: bool = False
    opacity: float = 1.0


class Profile(Strict):
    """A closed 2D outline in the component's local plan frame (inches)."""

    points: list[tuple[Length, Length]] = Field(min_length=3)
    axis_note: str | None = None
    src: Source


ComponentKind = Literal["column", "beam", "panel", "slab", "wall", "roof", "ring", "dome", "plate"]


class ComponentType(Strict):
    kind: ComponentKind
    material: str
    profile: str | None = None
    height: Length | None = None
    section: tuple[Length, Length] | None = Field(None, description="width, depth")
    thickness: Length | None = None
    facing: Length | None = Field(None, description="walls: masonry facing outside the dimensioned face")
    rise: Length | None = None
    collision: bool = True
    src: Source

    def wall_radii(self, r_face: float, to_m) -> tuple[float, float, float]:
        """(inside, outside, bearing centreline) radii in metres for a curved
        wall whose dimensioned face is at ``r_face`` metres: the wall's
        thickness runs inward from that face, any facing outward."""
        t = to_m(self.thickness)
        f = to_m(self.facing) if self.facing is not None else 0.0
        return r_face - t, r_face + f, r_face - t / 2

    @model_validator(mode="after")
    def _fields_for_kind(self) -> "ComponentType":
        need = {
            "column": ("profile", "height"),
            "beam": ("section",),
            "panel": ("thickness",),
            "slab": ("thickness",),
            "wall": ("thickness",),
            "roof": ("thickness",),
            "ring": ("section",),
            "dome": ("rise",),
            "plate": ("section",),
        }[self.kind]
        missing = [f for f in need if getattr(self, f) is None]
        if missing:
            raise ValueError(f"a {self.kind} type needs {', '.join(missing)}")
        return self


class Types(Strict):
    materials: dict[str, Material]
    profiles: dict[str, Profile] = {}
    components: dict[str, ComponentType]


# ---- radial pavilion pattern ------------------------------------------------


class PlanPolygon(Strict):
    sides: int = Field(ge=3, le=12)
    side: Length
    rotation: Angle = Field(description="angle of vertex 0, counter-clockwise from plan +X")
    src: Source


class TypeRef(Strict):
    type: str
    src: Source | None = None


class HeaderSpec(TypeRef):
    top: str = Field(description="level id of the top of header")


class StoneArcSpec(TypeRef):
    """A masonry arc about the pavilion's center point. Angles are plan
    angles, counter-clockwise from +X; the arc runs from ``start`` to ``end``."""

    radius_outside: Length
    start: Angle
    end: Angle
    top: str = Field(description="level id of the top of masonry")
    returns: bool = Field(False, description="add straight returns from the arc ends to the nearest columns")
    segments_per_90: int = Field(24, ge=4)


class SlabSpec(TypeRef):
    edge_offset: Length = Field(description="slab edge beyond the glazing line")


class GlazedRoofSpec(Strict):
    panel: str
    pitch: Angle
    src: Source


class ConicalRoofSpec(Strict):
    type: str
    eave_overhang: Length
    span: Literal["rest", "arcs"] = Field("rest", description="cover every non-glazed angle, or only the stone arcs")
    max_step: Angle = Field(7.5, description="largest angle between roof sections")
    nominal_pitch: Angle | None = Field(None, description="pitch stated in a general note, for cross-checking")
    src: Source


class SillSpec(TypeRef):
    pass


class RingSpec(TypeRef):
    top: str | None = Field(None, description="level id of the top of the compression ring")
    radius: Length | None = Field(None, description="outside radius, when it is given rather than derived")
    measured_radius: Length | None = Field(None, description="radius scaled off a sheet, for cross-checking")
    measured_src: Source | None = None

    @model_validator(mode="after")
    def _one_of(self) -> "RingSpec":
        if self.top is None and self.radius is None:
            raise ValueError("a ring needs its top level, its radius, or both")
        return self


class RafterSpec(TypeRef):
    cone_spacing: Angle
    glazed_midspan: bool = True


class RoofSpec(Strict):
    glazed: GlazedRoofSpec | None = None
    conical: ConicalRoofSpec
    sill: SillSpec
    rafters: RafterSpec
    ring: RingSpec
    skylight: TypeRef | None = None


def contiguous_run(sides: list[int], n: int) -> tuple[int, int]:
    """(first side, count) of a run of polygon sides that may wrap past n-1."""
    if not sides:
        return 0, 0
    s = sorted(set(sides))
    if len(s) == n:
        return 0, n
    for start in s:
        run = [(start + i) % n for i in range(len(s))]
        if sorted(run) == s:
            return start, len(s)
    raise ValueError(f"glazed sides {sides} are not one contiguous run")


class RadialPavilion(Strict):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    pattern: Literal["radial_pavilion"]
    name: str | None = None
    center: str
    floor: str
    plan: PlanPolygon
    glazed_sides: list[int] = []
    columns: TypeRef
    header: HeaderSpec
    doors: TypeRef
    stone_arcs: list[StoneArcSpec] = []
    slab: SlabSpec
    roof: RoofSpec | None = None

    @model_validator(mode="after")
    def _sides(self) -> "RadialPavilion":
        n = self.plan.sides
        bad = [s for s in self.glazed_sides if not 0 <= s < n]
        if bad:
            raise ValueError(f"glazed_sides {bad} outside 0..{n - 1}")
        contiguous_run(self.glazed_sides, n)
        if self.roof and self.roof.glazed and not self.glazed_sides:
            raise ValueError("a glazed roof needs glazed sides")
        return self


class StoneRing(Strict):
    """A free-standing masonry arc or ring: a stair tower, a hearth, a curved
    room wall that belongs to no hexagon."""

    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    pattern: Literal["stone_ring"]
    name: str | None = None
    center: str
    floor: str
    type: str
    radius_outside: Length
    start: Angle = 0.0
    end: Angle = 360.0
    top: str
    footprint: Literal["none", "sector", "circle"] = Field("sector", description="what this element adds to its floor slab")
    slab: TypeRef | None = None
    segments_per_90: int = Field(24, ge=4)
    src: Source


class RoundMass(Strict):
    """A solid cylinder of masonry: a flue or a round core."""

    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    pattern: Literal["round_mass"]
    name: str | None = None
    center: str
    floor: str
    type: str
    radius: Length
    top: str
    src: Source


class Link(Strict):
    """A straight glazed or masonry wall between two named points, such as
    ``pav_tl.v0`` (a pavilion vertex), ``pav_tl.arc0.start`` (an arc end) or a
    control point id."""

    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    pattern: Literal["link"]
    kind: Literal["glazed", "masonry"]
    start: str
    end: str
    floor: str
    type: str = Field(description="door panel type for glazed links, wall type for masonry")
    header: HeaderSpec | None = None
    top: str | None = Field(None, description="level id of the top of a masonry link")
    inset: Length = Fraction(6)
    lettered: Length | None = Field(None, description="length written on the sheet, for cross-checking")
    src: Source


Element = Annotated[RadialPavilion | StoneRing | RoundMass | Link, Field(discriminator="pattern")]


# ---- checks and open items --------------------------------------------------


class Chain(Strict):
    """A dimension string: its parts must sum to its overall."""

    sheet: str
    where: str
    parts: list[Length] = Field(min_length=2)
    overall: Length
    tolerance: Length = Fraction(1, 16)
    note: str | None = None


class RuleCheck(Strict):
    rule: Literal["hexagon_apothem", "hexagon_half_side"]
    side: Length
    dimensioned: Length
    sheet: str
    tolerance: Length = Fraction(1, 4)


class Checks(Strict):
    chains: list[Chain] = []
    rules: list[RuleCheck] = []


class OpenItem(Strict):
    kind: Literal["unresolved_reference", "missing_sheet", "assumption", "conflict", "simplification"]
    ref: str | None = None
    cited_by: list[str] = []
    what: str | None = None
    effect: str


class Floors(Strict):
    gap_fill: Length = Field(Fraction(0), description="close gaps narrower than twice this between footprints on one level")


class BuildingSpec(Strict):
    spec_version: str
    building: Building
    units: Literal["imperial", "metric"]
    datum: str = Field(description="level id that becomes z = 0 in Blender and Unreal")
    sheets: dict[str, Sheet]
    levels: dict[str, Level]
    control: dict[str, ControlPoint]
    types: Types
    elements: list[Element]
    floors: Floors = Floors()
    checks: Checks = Checks()
    open_items: list[OpenItem] = []

    @model_validator(mode="after")
    def _references(self) -> "BuildingSpec":
        errors: list[str] = []
        if self.datum not in self.levels:
            errors.append(f"datum: unknown level {self.datum!r}")
        comps, mats, profs = self.types.components, self.types.materials, self.types.profiles

        def src_ok(src: Source | None, where: str) -> None:
            if src and src.sheet and src.sheet not in self.sheets:
                errors.append(f"{where}: unknown sheet {src.sheet!r}")

        def level_ok(lvl: str | None, where: str) -> None:
            if lvl is not None and lvl not in self.levels:
                errors.append(f"{where}: unknown level {lvl!r}")

        def type_ok(ref: str | None, kind: str, where: str) -> None:
            if ref is None:
                return
            if ref not in comps:
                errors.append(f"{where}: unknown type {ref!r}")
            elif comps[ref].kind != kind:
                errors.append(f"{where}: type {ref!r} is a {comps[ref].kind}, expected a {kind}")

        for name, c in comps.items():
            if c.material not in mats:
                errors.append(f"types.components.{name}: unknown material {c.material!r}")
            if c.profile and c.profile not in profs:
                errors.append(f"types.components.{name}: unknown profile {c.profile!r}")
            src_ok(c.src, f"types.components.{name}")
        for name, lv in self.levels.items():
            src_ok(lv.src, f"levels.{name}")
        ids = [el.id for el in self.elements]
        for dup in sorted({i for i in ids if ids.count(i) > 1}):
            errors.append(f"elements: id {dup!r} used twice")
        for el in self.elements:
            w = f"elements.{el.id}"
            level_ok(el.floor, w)
            if isinstance(el, Link):
                type_ok(el.type, "panel" if el.kind == "glazed" else "wall", w)
                if el.header:
                    type_ok(el.header.type, "beam", w)
                    level_ok(el.header.top, w)
                level_ok(el.top, w)
                if el.kind == "masonry" and el.top is None:
                    errors.append(f"{w}: a masonry link needs a top level")
                continue
            if el.center not in self.control:
                errors.append(f"{w}: unknown control point {el.center!r}")
            if isinstance(el, StoneRing):
                type_ok(el.type, "wall", w)
                level_ok(el.top, w)
                if el.slab:
                    type_ok(el.slab.type, "slab", w)
                continue
            if isinstance(el, RoundMass):
                type_ok(el.type, "wall", w)
                level_ok(el.top, w)
                continue
            level_ok(el.header.top, w)
            type_ok(el.columns.type, "column", w)
            type_ok(el.header.type, "beam", w)
            type_ok(el.doors.type, "panel", w)
            type_ok(el.slab.type, "slab", w)
            for a in el.stone_arcs:
                type_ok(a.type, "wall", w)
                level_ok(a.top, w)
            if el.roof:
                r = el.roof
                level_ok(r.ring.top, w)
                type_ok(r.glazed.panel if r.glazed else None, "panel", w)
                type_ok(r.conical.type, "roof", w)
                type_ok(r.sill.type, "beam", w)
                type_ok(r.rafters.type, "beam", w)
                type_ok(r.ring.type, "ring", w)
                type_ok(r.skylight.type if r.skylight else None, "dome", w)
                if not r.glazed and (r.ring.top is None or r.ring.radius is None) and not el.stone_arcs:
                    errors.append(f"{w}: without a glazed roof the ring needs both its top and its radius, or a stone arc to bear on")
        for ch in self.checks.chains:
            if ch.sheet not in self.sheets:
                errors.append(f"checks.chains: unknown sheet {ch.sheet!r}")
        if errors:
            raise ValueError("; ".join(errors))
        return self


def json_schema() -> dict[str, Any]:
    return BuildingSpec.model_json_schema()
