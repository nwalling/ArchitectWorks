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
    rise: Length | None = None
    collision: bool = True
    src: Source

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
    rotation: Angle
    src: Source


class TypeRef(Strict):
    type: str
    src: Source | None = None


class HeaderSpec(TypeRef):
    top: str = Field(description="level id of the top of header")


class StoneWallSpec(TypeRef):
    radius_outside: Length
    top: str = Field(description="level id of the top of masonry")
    returns: bool = True
    segments: int = Field(48, ge=8)


class SlabSpec(TypeRef):
    edge_offset: Length = Field(description="slab edge beyond the glazing line")


class GlazedRoofSpec(Strict):
    panel: str
    pitch: Angle
    src: Source


class ConicalRoofSpec(Strict):
    type: str
    eave_overhang: Length
    segments: int = Field(24, ge=4)
    nominal_pitch: Angle | None = Field(None, description="pitch stated in a general note, for cross-checking")
    src: Source


class SillSpec(TypeRef):
    pass


class RingSpec(TypeRef):
    top: str = Field(description="level id of the top of the compression ring")
    measured_radius: Length | None = Field(None, description="radius scaled off a sheet, for cross-checking")
    measured_src: Source | None = None


class RafterSpec(TypeRef):
    cone_spacing: Angle
    glazed_midspan: bool = True


class RoofSpec(Strict):
    glazed: GlazedRoofSpec
    conical: ConicalRoofSpec
    sill: SillSpec
    rafters: RafterSpec
    ring: RingSpec
    skylight: TypeRef


class RadialPavilion(Strict):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    pattern: Literal["radial_pavilion"]
    name: str | None = None
    center: str
    floor: str
    plan: PlanPolygon
    glazed_sides: list[int]
    columns: TypeRef
    header: HeaderSpec
    doors: TypeRef
    stone_wall: StoneWallSpec
    slab: SlabSpec
    roof: RoofSpec

    @model_validator(mode="after")
    def _sides(self) -> "RadialPavilion":
        n = self.plan.sides
        bad = [s for s in self.glazed_sides if not 0 <= s < n]
        if bad:
            raise ValueError(f"glazed_sides {bad} outside 0..{n - 1}")
        g = sorted(self.glazed_sides)
        if g != list(range(g[0], g[0] + len(g))):
            raise ValueError("glazed_sides must be one contiguous run (phase 0)")
        return self


Element = Annotated[RadialPavilion, Field(discriminator="pattern")]


# ---- checks and open items --------------------------------------------------


class Chain(Strict):
    """A dimension string: its parts must sum to its overall."""

    sheet: str
    where: str
    parts: list[Length] = Field(min_length=2)
    overall: Length
    tolerance: Length = Fraction(1, 16)


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

        for name, c in comps.items():
            if c.material not in mats:
                errors.append(f"types.components.{name}: unknown material {c.material!r}")
            if c.profile and c.profile not in profs:
                errors.append(f"types.components.{name}: unknown profile {c.profile!r}")
            src_ok(c.src, f"types.components.{name}")
        for name, lv in self.levels.items():
            src_ok(lv.src, f"levels.{name}")
        for el in self.elements:
            w = f"elements.{el.id}"
            if el.center not in self.control:
                errors.append(f"{w}: unknown control point {el.center!r}")
            for lvl in (el.floor, el.header.top, el.stone_wall.top, el.roof.ring.top):
                if lvl not in self.levels:
                    errors.append(f"{w}: unknown level {lvl!r}")
            for ref, kind in (
                (el.columns.type, "column"),
                (el.header.type, "beam"),
                (el.doors.type, "panel"),
                (el.stone_wall.type, "wall"),
                (el.slab.type, "slab"),
                (el.roof.glazed.panel, "panel"),
                (el.roof.conical.type, "roof"),
                (el.roof.sill.type, "beam"),
                (el.roof.rafters.type, "beam"),
                (el.roof.ring.type, "ring"),
                (el.roof.skylight.type, "dome"),
            ):
                if ref not in comps:
                    errors.append(f"{w}: unknown type {ref!r}")
                elif comps[ref].kind != kind:
                    errors.append(f"{w}: type {ref!r} is a {comps[ref].kind}, expected a {kind}")
        for ch in self.checks.chains:
            if ch.sheet not in self.sheets:
                errors.append(f"checks.chains: unknown sheet {ch.sheet!r}")
        if errors:
            raise ValueError("; ".join(errors))
        return self


def json_schema() -> dict[str, Any]:
    return BuildingSpec.model_json_schema()
