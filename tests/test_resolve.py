import json
import math
from collections import Counter

import pytest

from architectworks.geom import mesh as g
from architectworks.resolve import resolve
from architectworks.units import parse_length, to_metres


def mesh_of(d):
    return g.Mesh([tuple(v) for v in d["verts"]], [tuple(f) for f in d["faces"]], d["face_mat"])


def test_counts(resolved):
    by = Counter(p["component"] for p in resolved["placements"])
    names = {c["name"] for c in resolved["components"]}
    assert by["SM_EH_ColumnTyp_A"] == 4  # the four C.P.s bounding three glazed bays
    assert by["SM_EH_Header6x6_A"] == 3
    assert by["SM_EH_DoorGlass_A"] == 3
    assert by["SM_EH_RoofGlass_A"] == 3
    assert by["SM_EH_WallStoneReturn_A"] == 2
    rafters = sum(n for c, n in by.items() if c.startswith("SM_EH_Rafter4x6"))
    assert rafters == 4 + 3 + 23  # hips, midspans, cone at 7.5 degrees over 180
    assert "SM_EH_AW_HandednessMarker" in names


def test_every_component_is_a_valid_solid(resolved):
    for c in resolved["components"]:
        m = mesh_of(c["mesh"])
        assert g.is_closed(m), c["name"]
        assert g.volume(m) > 0, c["name"]
        assert g.is_planar(m, tol=1e-5), c["name"]
        assert len(c["mesh"]["uv"]) == len(c["mesh"]["faces"])


def test_every_collision_piece_is_convex(resolved):
    for c in resolved["components"]:
        for i, h in enumerate(c["collision"]):
            assert g.is_convex(mesh_of(h), tol=1e-5), f"{c['name']} hull {i}"


def test_translucent_meshes_are_not_nanite(resolved):
    flags = {c["name"]: c["nanite"] for c in resolved["components"]}
    assert flags["SM_EH_DoorGlass_A"] is False
    assert flags["SM_EH_RoofGlass_A"] is False
    assert flags["SM_EH_SkylightDome_A"] is False
    assert flags["SM_EH_ColumnTyp_A"] is True


def test_columns_stand_on_the_hexagon(resolved):
    side = to_metres(parse_length("8'-10\""))
    cols = [p for p in resolved["placements"] if p["component"] == "SM_EH_ColumnTyp_A"]
    for p in cols:
        m = p["matrix"]
        assert math.hypot(m[0][3], m[1][3]) == pytest.approx(side, abs=1e-9)
        assert m[2][3] == pytest.approx(0.0)
        # long axis of the profile points out along the radius
        y_axis = (m[0][1], m[1][1])
        r = (m[0][3] / side, m[1][3] / side)
        assert y_axis[0] * r[0] + y_axis[1] * r[1] == pytest.approx(1.0)


def test_header_sits_on_columns(resolved):
    col = next(c for c in resolved["components"] if c["name"] == "SM_EH_ColumnTyp_A")
    hdr = [p for p in resolved["placements"] if p["component"] == "SM_EH_Header6x6_A"]
    col_top = col["bounds"]["max"][2]
    for p in hdr:
        top = p["matrix"][2][3]
        assert top - to_metres(parse_length("6\"")) == pytest.approx(col_top, abs=1e-9)
        assert top == pytest.approx(to_metres(parse_length("7'-1 1/2\"")), abs=1e-9)


def test_derived_values(resolved):
    d = resolved["derived"]["pav_a"]
    assert d["ring_radius"] == "2'-8 7/8\""
    assert d["hip_pitch_deg"] == pytest.approx(26.565, abs=1e-3)
    assert d["conical_pitch_deg"] == pytest.approx(27.776, abs=1e-3)
    assert d["door_inset_from_column_cp"] == "5 13/16\""


def test_checks(resolved):
    status = {c["id"]: c["status"] for c in resolved["checks"]}
    assert all(status[f"chain.foundation.{i}"] == "pass" for i in range(5))
    assert all(v == "pass" for k, v in status.items() if k.startswith("rule."))
    assert status["pav_a.header_depth"] == "pass"
    # Real disagreements between sheets are reported, not hidden.
    assert status["pav_a.ring_radius"] == "conflict"
    assert status["pav_a.conical_pitch"] == "conflict"
    assert status["pav_a.hip_step"] == "conflict"


def test_assumed_elements_are_marked(resolved):
    assumed = {p["id"] for p in resolved["placements"] if p["basis"] == "assumed"}
    assert {"pav_a.wall.return.start", "pav_a.wall.return.end", "pav_a.slab"} <= assumed


def test_resolution_is_deterministic(spec, resolved):
    again = resolve(spec)
    assert json.dumps(again, sort_keys=True) == json.dumps(resolved, sort_keys=True)


def test_building_size(resolved):
    # The stone side reaches the 11'-8" wall plus a 1'-0" eave; the glazed
    # side reaches the column faces. Across X the roof spans 2 x 12'-8".
    pts = []
    comps = {c["name"]: c for c in resolved["components"]}
    from architectworks.geom.xform import transformed_box_bounds

    for p in resolved["placements"]:
        if p["category"] == "Fixtures":
            continue
        c = comps[p["component"]]
        pts.extend(transformed_box_bounds(tuple(map(tuple, p["matrix"])), tuple(c["bounds"]["min"]), tuple(c["bounds"]["max"])))
    xs = [p[0] for p in pts]
    assert max(xs) - min(xs) == pytest.approx(2 * to_metres(parse_length("12'-8\"")), abs=1e-3)
