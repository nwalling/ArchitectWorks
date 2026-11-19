"""What the reader returns for a plan sheet.

Positions are full-sheet pixels of the original photo. Where a detected mark
sits on the thing, its id is given too and the mark's position wins. Lengths
are copied as lettered ("7'-7 3/4\\""). Unknown text is "" and an unknown
mark is "", so every field is always present (the API's strict structured
outputs want that).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Confidence = Literal["high", "medium", "low"]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SheetInfo(Strict):
    kind: Literal["plan", "elevation", "section", "detail", "other"]
    title: str
    scale: str
    rotation_deg: Literal[0, 90, 180, 270] = Field(description="clockwise rotation that makes the lettering upright")


class Point(Strict):
    x: float
    y: float


class CenterPoint(Strict):
    label: str = Field(description="as lettered, e.g. 'C.P. 3', or a short description")
    mark: str
    x: float
    y: float
    confidence: Confidence


class Column(Strict):
    mark: str
    x: float
    y: float
    center: str = Field(description="label of the center point this column belongs to")
    confidence: Confidence


class ChainReading(Strict):
    """Consecutive dimensions along one dimension line. Ticks are listed left to
    right for horizontal chains and top to bottom for vertical ones; there is
    one more tick than there are parts."""

    side: Literal["top", "bottom", "left", "right", "interior"]
    axis: Literal["horizontal", "vertical"]
    ticks: list[Point]
    parts: list[str]
    overall: str
    confidence: Confidence


class Note(Strict):
    text: str
    kind: Literal["typical_side", "radius", "level", "general", "other"]
    x: float
    y: float
    applies_to: list[str] = Field(description="labels of the center points the note governs")


class Unreadable(Strict):
    x: float
    y: float
    note: str


class PlanReading(Strict):
    sheet: SheetInfo
    center_points: list[CenterPoint]
    columns: list[Column]
    chains: list[ChainReading]
    notes: list[Note]
    unreadable: list[Unreadable]


def _strict_schema(node):
    """Inline $refs and mark every object closed, for structured outputs."""
    defs = node.get("$defs", {})

    def walk(n):
        if isinstance(n, dict):
            if "$ref" in n:
                return walk(defs[n["$ref"].split("/")[-1]])
            out = {k: walk(v) for k, v in n.items() if k not in ("$defs", "title", "description", "default")}
            if out.get("type") == "object":
                out["additionalProperties"] = False
                out["required"] = list(out.get("properties", {}).keys())
            return out
        if isinstance(n, list):
            return [walk(v) for v in n]
        return n

    return walk(node)


def plan_reading_schema() -> dict:
    return _strict_schema(PlanReading.model_json_schema())


def load_reading(path: str | Path) -> PlanReading:
    """Read a reading file; top-level keys starting with "_" are comments."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return PlanReading.model_validate({k: v for k, v in data.items() if not k.startswith("_")})
