"""Stage 2b: the reader. Claude looks at the sheet and names what detection found.

The model gets the whole sheet once, the list of candidate marks, and two
tools for looking closer. It answers with a ``PlanReading`` (structured
output, so the JSON always parses). Positions it gives are checked and
replaced by detected marks downstream; what we want from it is meaning:
which mark is a center point, which numbers belong to which dimension line.

Every API response is recorded on disk, keyed by a hash of the request. A
second run with the same sheet, detections and prompt replays the recording
instead of calling the API, so tests, scoring and draft-building can be
rerun for free and a reading can be reproduced exactly.
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from pydantic import BaseModel, Field, ValidationError

from architectworks.reader.detect import Detections
from architectworks.reader.schema import PlanReading, plan_reading_schema
from architectworks.reader.views import View, render

MODEL = "claude-opus-5-5"
EFFORT = "high"
BETAS = ["server-side-fallback-2026-07-01"]
MAX_TOKENS = 64000
MAX_TURNS = 40
MAX_MARKS_PER_CALL = 8

SYSTEM = """\
You are reading one photographed sheet of a hand-drafted architectural drawing set so that the building can be modelled from it.

Image processing has already found candidate marks on the sheet and numbered them. M ids are compact dark blobs, often column marks. C ids are points where many lines at different angles meet, often the center points that radial plans are set out from. Where a mark sits on the thing you are naming, give its id: its position is measured more precisely than anyone can point. Give your own x and y as well, always in full-sheet pixels. Many marks are noise (lettering, hatching, crossings); ignore those.

What to report (the response format lists every field):
- sheet: kind, the title as lettered, the scale as lettered, and the clockwise rotation that makes the lettering upright.
- center_points: every center point the plan is set out from (lettered "C.P." or plain from the radial lines), each with a short label that says which part of the building it governs.
- columns: column marks, each with the label of the center point whose polygon it sits on.
- chains: every run of consecutive dimensions along one dimension line. List the ticks in order (left to right for horizontal lines, top to bottom for vertical ones), at the tick marks themselves where the extension lines cross the dimension line; there is one more tick than there are parts. Copy every dimension exactly as lettered, for example 7'-7 3/4". Give the overall dimension if the chain has one, else "".
- notes: notes that govern geometry: typical side lengths, radii, levels. applies_to lists the labels of the center points a note governs.
- unreadable: anything that matters but that you cannot read with confidence, with where it is.

How to work: the first image is the whole sheet, too small to read lettering reliably. Use view_region to zoom anywhere (its grid labels are sheet pixels at every zoom) and view_marks to look closely at numbered marks. Read every dimension you report from a zoomed view, and where a chain has an overall dimension, check that its parts add up to it. Do not guess to fill a field: put what you cannot read under unreadable, or use confidence "low".
"""


class ViewRegionInput(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float
    marks: bool = Field(True, description="draw the numbered candidate marks")


class ViewMarksInput(BaseModel):
    ids: list[str]
    radius: float = 60.0


TOOLS: list[dict[str, Any]] = [
    {
        "name": "view_region",
        "description": (
            "Show a rectangle of the sheet, given in full-sheet pixels, enlarged up to 4x. "
            "Grid lines are labelled in sheet pixels. With marks on, candidate marks in the "
            "rectangle are circled with their ids and listed under the image."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "x0": {"type": "number"},
                "y0": {"type": "number"},
                "x1": {"type": "number"},
                "y1": {"type": "number"},
                "marks": {"type": "boolean"},
            },
            "required": ["x0", "y0", "x1", "y1"],
        },
        "eager_input_streaming": True,
    },
    {
        "name": "view_marks",
        "description": (
            f"Show a close-up around each listed mark (up to {MAX_MARKS_PER_CALL} per call), "
            "radius in sheet pixels (default 60), with nearby marks circled."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ids": {"type": "array", "items": {"type": "string"}},
                "radius": {"type": "number"},
            },
            "required": ["ids"],
        },
        "eager_input_streaming": True,
    },
]


class ReaderError(RuntimeError):
    pass


class ReaderRefused(ReaderError):
    pass


class NoRecording(ReaderError):
    pass


def _image_block(view: View) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64.standard_b64encode(view.png).decode("ascii")}}


def marks_listing(det: Detections) -> str:
    rows = [f"{m.id} {m.kind} ({m.x:.0f},{m.y:.0f})" for m in det.marks]
    return f"Sheet is {det.width} x {det.height} pixels. {len(det.marks)} candidate marks:\n" + "\n".join(rows)


class Recorder:
    """Request-hash keyed recordings of API responses."""

    def __init__(self, directory: Path | None, offline: bool = False) -> None:
        self.directory = Path(directory) if directory else None
        self.offline = offline
        self.hits = 0
        self.calls = 0

    @staticmethod
    def key(params: dict) -> str:
        blob = json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]

    def get(self, params: dict) -> dict | None:
        if self.directory is None:
            return None
        path = self.directory / f"{self.key(params)}.json"
        if path.exists():
            self.hits += 1
            return json.loads(path.read_text(encoding="utf-8"))
        return None

    def put(self, params: dict, response: dict) -> None:
        if self.directory is None:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / f"{self.key(params)}.json").write_text(json.dumps(response, indent=1) + "\n", encoding="utf-8")


def _call(client, params: dict) -> dict:
    with client.beta.messages.stream(**params) as stream:
        message = stream.get_final_message()
    return message.to_dict(exclude_none=True)


def _validated(model: type[BaseModel], raw: Any) -> BaseModel:
    if not isinstance(raw, dict):
        raw = {}
    return model.model_validate(raw)


@dataclass
class ReadResult:
    reading: PlanReading
    turns: int
    views: int
    usage: dict[str, int] = field(default_factory=dict)
    served_by: list[str] = field(default_factory=list)
    replayed: int = 0
    called: int = 0

    def summary(self) -> dict:
        return {"turns": self.turns, "views": self.views, "usage": self.usage, "served_by": self.served_by, "replayed": self.replayed, "called": self.called}


def read_plan(
    flat: np.ndarray,
    det: Detections,
    client=None,
    recordings: Path | None = None,
    offline: bool = False,
    max_turns: int = MAX_TURNS,
    log: Callable[[str], None] = print,
) -> ReadResult:
    """Run the reader on one plan sheet and return its reading.

    ``client`` is an ``anthropic.Anthropic``; one is made from the
    environment (ANTHROPIC_API_KEY) if needed and none is given. With
    ``offline`` only recordings are used and a missing one raises
    ``NoRecording``.
    """
    rec = Recorder(recordings, offline)
    h, w = flat.shape
    overview = render(flat, 0, 0, w, h)
    messages: list[dict] = [
        {
            "role": "user",
            "content": [
                _image_block(overview),
                {"type": "text", "text": "The whole sheet. " + overview.caption()},
                {"type": "text", "text": marks_listing(det)},
                {"type": "text", "text": "Read this sheet and report what it shows."},
            ],
        }
    ]
    by_id = det.by_id()
    usage: dict[str, int] = {}
    served: list[str] = []
    views = 0

    def run_tool(name: str, raw: Any) -> list[dict]:
        nonlocal views
        if name == "view_region":
            a = _validated(ViewRegionInput, raw)
            v = render(flat, min(a.x0, a.x1), min(a.y0, a.y1), max(a.x0, a.x1), max(a.y0, a.y1), det if a.marks else None)
            views += 1
            return [_image_block(v), {"type": "text", "text": v.caption()}]
        if name == "view_marks":
            a = _validated(ViewMarksInput, raw)
            if not a.ids:
                raise ValueError("no mark ids given")
            if len(a.ids) > MAX_MARKS_PER_CALL:
                raise ValueError(f"at most {MAX_MARKS_PER_CALL} marks per call")
            unknown = [i for i in a.ids if i not in by_id]
            if unknown:
                raise ValueError("unknown mark ids: " + ", ".join(unknown))
            r = min(max(a.radius, 15.0), 400.0)
            out: list[dict] = []
            for i in a.ids:
                m = by_id[i]
                v = render(flat, m.x - r, m.y - r, m.x + r, m.y + r, det)
                views += 1
                out += [_image_block(v), {"type": "text", "text": f"Around {i}. " + v.caption()}]
            return out
        raise ValueError(f"no tool named {name}")

    for turn in range(1, max_turns + 1):
        last = turn == max_turns
        params: dict[str, Any] = {
            "model": MODEL,
            "max_tokens": MAX_TOKENS,
            "system": SYSTEM,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": EFFORT, "format": {"type": "json_schema", "schema": plan_reading_schema()}},
            "tools": TOOLS,
            "tool_choice": {"type": "none"} if last else {"type": "auto"},
            "cache_control": {"type": "ephemeral"},
            "betas": BETAS,
            "fallbacks": "default",
            "messages": messages,
        }
        resp = rec.get(params)
        if resp is None:
            if offline:
                raise NoRecording(f"turn {turn}: no recorded response for this request in {recordings}")
            if client is None:
                import anthropic

                client = anthropic.Anthropic()
            resp = _call(client, params)
            rec.calls += 1
            rec.put(params, resp)
        for k, v in (resp.get("usage") or {}).items():
            if isinstance(v, int):
                usage[k] = usage.get(k, 0) + v
        if resp.get("model") and resp["model"] not in served:
            served.append(resp["model"])

        stop = resp.get("stop_reason")
        content = resp.get("content", [])
        if stop == "refusal":
            details = resp.get("stop_details") or {}
            raise ReaderRefused(f"the reader declined (category {details.get('category')}): {details.get('explanation', '')}")
        if stop == "max_tokens":
            raise ReaderError(f"turn {turn}: the reply hit max_tokens ({MAX_TOKENS})")
        messages.append({"role": "assistant", "content": content})
        if stop == "pause_turn":
            continue
        if stop == "tool_use":
            results: list[dict] = []
            for block in content:
                if block.get("type") != "tool_use":
                    continue
                try:
                    body = run_tool(block["name"], block.get("input"))
                    results.append({"type": "tool_result", "tool_use_id": block["id"], "content": body})
                except (ValidationError, ValueError) as err:
                    results.append({"type": "tool_result", "tool_use_id": block["id"], "content": f"error: {err}", "is_error": True})
            if turn + 1 == max_turns:
                results.append({"type": "text", "text": "That is the last view. Give your reading now."})
            messages.append({"role": "user", "content": results})
            log(f"  turn {turn}: {sum(1 for b in content if b.get('type') == 'tool_use')} tool calls, {views} views so far")
            continue
        texts = [b["text"] for b in content if b.get("type") == "text"]
        if not texts:
            raise ReaderError(f"turn {turn}: stopped ({stop}) without a reading")
        try:
            reading = PlanReading.model_validate_json(texts[-1])
        except ValidationError as err:
            raise ReaderError(f"turn {turn}: the reading does not validate: {err}") from err
        return ReadResult(reading, turn, views, usage, served, rec.hits, rec.calls)
    raise ReaderError(f"no reading after {max_turns} turns")
