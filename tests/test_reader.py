"""Phase 2: detection, the draft builder, the scorer, and the reader's plumbing.

The reader itself needs an API key; here it runs against a scripted fake
client, which checks the request shapes, the tool loop and the recordings.
How well Claude actually reads the sheet is measured with ``aw read`` and
``aw draft --score`` once a key is available.
"""

import json
from fractions import Fraction

import pytest
import yaml

cv2 = pytest.importorskip("cv2")
pytest.importorskip("scipy")

from architectworks.reader import claude  # noqa: E402
from architectworks.reader.detect import detect, load_flat, nearest  # noqa: E402
from architectworks.reader.draft import build_draft  # noqa: E402
from architectworks.reader.schema import PlanReading, load_reading, plan_reading_schema  # noqa: E402
from architectworks.reader.score import score  # noqa: E402
from architectworks.reader.views import render  # noqa: E402
from architectworks.spec import load_spec  # noqa: E402
from tests.conftest import ROOT  # noqa: E402

PROJECT = ROOT / "projects" / "example-house"
SHEET = PROJECT / "sheets" / "foundation-plan.jpg"
CALIB = PROJECT / "sheets" / "foundation-plan.calib.yaml"
HAND = PROJECT / "readings" / "foundation-plan.hand.json"


@pytest.fixture(scope="module")
def flat():
    return load_flat(SHEET)


@pytest.fixture(scope="module")
def det(flat):
    return detect(flat)


@pytest.fixture(scope="module")
def hand():
    return load_reading(HAND)


@pytest.fixture(scope="module")
def key():
    return load_spec(PROJECT / "house.yaml")


@pytest.fixture(scope="module")
def draft(hand, flat, det):
    return build_draft(hand, flat, det).to_json()


# ---- detection ----


def test_detection_finds_the_calibration_points(det):
    points = yaml.safe_load(CALIB.read_text())["points"]
    found = {name: nearest(det, x, y)[1] for name, (x, y) in points.items()}
    near = [n for n, d in found.items() if d <= 4.0]
    assert len(near) >= len(points) - 1, {n: round(d, 1) for n, d in found.items() if d > 4.0}


def test_mark_ids_are_unique_and_in_reading_order(det):
    ids = [m.id for m in det.marks]
    assert len(ids) == len(set(ids))
    blobs = [m for m in det.marks if m.kind == "blob"]
    assert [round(m.y / 40) for m in blobs] == sorted(round(m.y / 40) for m in blobs)


def test_hand_reading_marks_exist(det, hand):
    ids = det.by_id()
    used = [cp.mark for cp in hand.center_points if cp.mark] + [c.mark for c in hand.columns if c.mark]
    assert used and all(m in ids for m in used)


# ---- views ----


def test_view_reports_its_region_and_marks(flat, det):
    v = render(flat, 400, 300, 700, 600, det)
    assert v.png[:8] == b"\x89PNG\r\n\x1a\n"
    assert (v.x0, v.y0, v.x1, v.y1) == (400, 300, 700, 600)
    assert v.scale == pytest.approx(4.0)
    assert v.marks and all(400 <= m["x"] <= 700 and 300 <= m["y"] <= 600 for m in v.marks)
    assert "Sheet pixels x 400..700" in v.caption()
    whole = render(flat, 0, 0, flat.shape[1], flat.shape[0])
    assert max(whole.x1 - whole.x0, whole.y1 - whole.y0) * whole.scale <= 1568 + 1
    with pytest.raises(ValueError):
        render(flat, 10, 10, 12, 12)


# ---- schema ----


def test_reading_schema_is_strict_and_inlined():
    schema = plan_reading_schema()
    text = json.dumps(schema)
    assert "$ref" not in text and "$defs" not in text

    def walk(n):
        if isinstance(n, dict):
            if n.get("type") == "object":
                assert n["additionalProperties"] is False
                assert sorted(n["required"]) == sorted(n["properties"])
            for v in n.values():
                walk(v)
        elif isinstance(n, list):
            for v in n:
                walk(v)

    walk(schema)


def test_hand_reading_validates(hand):
    assert len(hand.center_points) == 9 and len(hand.chains) == 6
    for ch in hand.chains:
        assert len(ch.ticks) == len(ch.parts) + 1


# ---- draft ----


def test_draft_from_the_hand_reading(draft):
    assert draft["fit"]["dimensioned_both_axes"] >= 8
    assert draft["fit"]["feature_residual_in"]["mean"] < 2.0
    assert len(draft["control"]) == 9
    by_label = {p["center"]: p for p in draft["pavilions"]}
    nw = by_label["C.P. north-west pavilion"]
    assert (nw["sides"], nw["side"], nw["rotation_deg"]) == (6, "8'-10\"", 0)
    assert sorted(nw["glazed_sides"]) == [0, 1, 2]
    ne = by_label["C.P. north-east pavilion"]
    assert ne["rotation_deg"] == 30 and sorted(ne["glazed_sides"]) == [0, 1, 5]
    # anything the draft could not pin down exactly is flagged for review
    assert any("measured" in r for r in draft["review"])


# ---- score ----


def test_score_of_the_hand_reading_draft(draft, key):
    s = score(draft, key)
    summary = s["summary"]
    assert summary["center_points"]["matched"] == 9
    assert summary["center_points"]["exact"] >= 3
    assert summary["center_points"]["within_2_in"] >= 6
    assert summary["pavilions"]["all_checks_pass"] >= 4
    assert summary["score"] >= 0.8
    # every center point the key dimensions is within 2 in
    for p in s["points"]:
        if p["key_basis"] == "dimensioned":
            assert p["close"], p


def test_score_is_translation_free_and_perfect_on_the_key_itself(key):
    from architectworks.units import format_length

    shift = (Fraction(149, 4), Fraction(-203, 2))  # 3'-1 1/4", -8'-5 1/2"
    control = {cid: {"x": format_length(cp.at[0] - shift[0]), "y": format_length(cp.at[1] - shift[1]), "x_basis": "dimensioned", "y_basis": "dimensioned"} for cid, cp in key.control.items()}
    pavilions = [
        {"center": el.center, "sides": el.plan.sides, "side": format_length(el.plan.side), "rotation_deg": float(el.plan.rotation), "glazed_sides": list(el.glazed_sides)}
        for el in key.elements
        if el.pattern == "radial_pavilion"
    ]
    s = score({"control": control, "pavilions": pavilions}, key)
    assert s["translation_in"] == [float(shift[0]), float(shift[1])]
    assert s["summary"]["score"] == 1.0
    assert s["summary"]["center_points"]["exact"] == len(key.control)

    # a wrong side length and a missing point are both counted
    pavilions[0]["side"] = "9'-0\""
    first = next(iter(control))
    del control[first]
    s = score({"control": control, "pavilions": pavilions}, key)
    assert s["summary"]["center_points"]["missed"] == [first]
    assert s["summary"]["score"] < 1.0


# ---- the reader, against a scripted client ----


class FakeMessage:
    def __init__(self, data):
        self.data = data

    def to_dict(self, exclude_none=True):
        return json.loads(json.dumps(self.data))


class FakeStream:
    def __init__(self, data):
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return FakeMessage(self.data)


class FakeClient:
    """Plays back scripted replies and keeps every request it was sent."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        outer = self

        class _Messages:
            def stream(self, **params):
                outer.requests.append(json.loads(json.dumps(params)))
                return FakeStream(outer.replies.pop(0))

        class _Beta:
            messages = _Messages()

        self.beta = _Beta()


def reply(content, stop, model=claude.MODEL):
    return {"id": "msg_fake", "type": "message", "role": "assistant", "model": model, "content": content, "stop_reason": stop, "usage": {"input_tokens": 1000, "output_tokens": 200, "cache_read_input_tokens": 500}}


def scripted(hand):
    final = json.dumps(hand.model_dump())
    return [
        reply(
            [
                {"type": "thinking", "thinking": "", "signature": "sig1"},
                {"type": "tool_use", "id": "tu1", "name": "view_region", "input": {"x0": 300, "y0": 150, "x1": 900, "y1": 260}},
                {"type": "tool_use", "id": "tu2", "name": "view_marks", "input": {"ids": ["C38", "NOPE"]}},
            ],
            "tool_use",
        ),
        reply([{"type": "tool_use", "id": "tu3", "name": "view_marks", "input": {"ids": ["C38", "M56"], "radius": 40}}], "tool_use"),
        reply([{"type": "thinking", "thinking": "", "signature": "sig2"}, {"type": "text", "text": final}], "end_turn"),
    ]


def test_reader_tool_loop_and_recordings(tmp_path, flat, det, hand):
    client = FakeClient(scripted(hand))
    rec = tmp_path / "recordings"
    result = claude.read_plan(flat, det, client=client, recordings=rec, log=lambda _: None)
    assert result.reading == hand
    assert (result.turns, result.views, result.called, result.replayed) == (3, 3, 3, 0)
    assert result.usage["input_tokens"] == 3000 and result.usage["cache_read_input_tokens"] == 1500

    first = client.requests[0]
    assert first["model"] == "claude-opus-5-5"
    assert first["output_config"]["effort"] == "high"
    assert first["output_config"]["format"]["schema"] == plan_reading_schema()
    assert first["betas"] == ["server-side-fallback-2026-07-01"] and first["fallbacks"] == "default"
    assert first["cache_control"] == {"type": "ephemeral"}
    assert first["tool_choice"] == {"type": "auto"}
    assert "budget_tokens" not in json.dumps(first["thinking"])
    opening = first["messages"][0]["content"]
    assert opening[0]["type"] == "image" and "C38 convergence" in opening[2]["text"]

    # turn 2 carries turn 1 back unchanged, then the tool results in one user message
    second = client.requests[1]["messages"]
    assert second[1] == {"role": "assistant", "content": scripted(hand)[0]["content"]}
    results = second[2]["content"]
    assert [r["tool_use_id"] for r in results] == ["tu1", "tu2"]
    assert results[0]["content"][0]["type"] == "image" and "Sheet pixels" in results[0]["content"][1]["text"]
    assert results[1]["is_error"] is True and "NOPE" in results[1]["content"]
    third = client.requests[2]["messages"][-1]["content"]
    assert [b["type"] for b in third[0]["content"]] == ["image", "text", "image", "text"]

    # a second run replays from the recordings without calling the API
    again = claude.read_plan(flat, det, client=FakeClient([]), recordings=rec, offline=True, log=lambda _: None)
    assert again.reading == hand and (again.called, again.replayed) == (0, 3)
    with pytest.raises(claude.NoRecording):
        claude.read_plan(flat, det, recordings=tmp_path / "empty", offline=True, log=lambda _: None)


def test_reader_last_turn_and_failures(tmp_path, flat, det, hand):
    replies = scripted(hand)
    client = FakeClient([replies[0], replies[2]])
    result = claude.read_plan(flat, det, client=client, max_turns=2, log=lambda _: None)
    assert result.turns == 2
    last = client.requests[1]
    assert last["tool_choice"] == {"type": "none"}
    assert last["messages"][-1]["content"][-1] == {"type": "text", "text": "That is the last view. Give your reading now."}

    refusal = reply([], "refusal")
    refusal["stop_details"] = {"type": "refusal", "category": None, "explanation": "test"}
    with pytest.raises(claude.ReaderRefused):
        claude.read_plan(flat, det, client=FakeClient([refusal]), log=lambda _: None)
    with pytest.raises(claude.ReaderError):
        claude.read_plan(flat, det, client=FakeClient([reply([{"type": "text", "text": "{}"}], "end_turn")]), log=lambda _: None)
    with pytest.raises(claude.ReaderError):
        claude.read_plan(flat, det, client=FakeClient([reply([], "max_tokens")]), log=lambda _: None)
