"""Phase 1: the whole example house, and its agreement with the foundation plan."""

import importlib.util
import math
from collections import Counter
from pathlib import Path

import pytest

from architectworks.geom import mesh as g
from architectworks.resolve import resolve
from architectworks.spec import load_spec
from tests.conftest import ROOT

HOUSE = ROOT / "projects" / "example-house" / "house.yaml"


@pytest.fixture(scope="module")
def house_spec():
    return load_spec(HOUSE)


@pytest.fixture(scope="module")
def house(house_spec):
    return resolve(house_spec)


def mesh_of(d):
    return g.Mesh([tuple(v) for v in d["verts"]], [tuple(f) for f in d["faces"]], d["face_mat"])


def test_every_element_resolves(house):
    elements = Counter(p["element"] for p in house["placements"])
    for eid in ("pav_tl", "pav_tr", "pav_center", "pav_br", "pav_living", "hearth", "music", "tower_stair", "core_n", "core_s", "floors"):
        assert elements[eid] > 0, eid
    assert sum(1 for p in house["placements"] if p["element"].startswith("link_")) == 10


def test_every_component_is_valid(house):
    for c in house["components"]:
        m = mesh_of(c["mesh"])
        assert g.is_closed(m) and g.volume(m) > 0, c["name"]
        for i, h in enumerate(c["collision"]):
            assert g.is_convex(mesh_of(h), tol=1e-5), f"{c['name']} hull {i}"


def test_all_dimension_chains_close(house):
    chains = [c for c in house["checks"] if c["kind"] == "chain"]
    assert len(chains) == 10
    assert all(c["status"] == "pass" for c in chains), [c["message"] for c in chains if c["status"] != "pass"]


def test_lettered_link_length_agrees_with_geometry(house):
    # The 5'-0" between the two north pavilions follows from their centers and sides alone.
    link = next(c for c in house["checks"] if c["id"] == "link_tl_tr.length")
    assert link["status"] == "pass"
    assert abs(link["values"]["difference_in"]) < 0.25


def test_columns_land_on_dimensioned_lines(house):
    pts = house["points"]
    ft = 0.3048
    assert pts["pav_tl.v0"][0] == pytest.approx(14.5 * ft, abs=1e-6)  # top chain 14'-6"
    assert pts["pav_tr.v2"][0] == pytest.approx((19 + 4 / 12) * ft, abs=2e-3)  # top chain 19'-4"
    assert pts["pav_center.v2"][0] == pytest.approx((7 + 7.75 / 12) * ft, abs=2e-3)  # bottom chain 7'-7 3/4"


def test_floors_per_level(house):
    floors = house["derived"]["floors"]
    assert set(floors) == {"ff_bedroom", "ff_music"}
    music = next(p for p in house["placements"] if p["id"] == "slab.ff_music")
    assert music["matrix"][2][3] == pytest.approx(-(1 + 8 / 12) * 0.3048, abs=1e-6)  # 98'-4" is 1'-8" below the datum


@pytest.mark.skipif(importlib.util.find_spec("cv2") is None, reason="needs the sheets extra (OpenCV)")
def test_model_agrees_with_the_foundation_plan(house, tmp_path):
    from architectworks import sheets

    calib_path = ROOT / "projects" / "example-house" / "sheets" / "foundation-plan.calib.yaml"
    calib = sheets.fit(calib_path, house["points"])
    assert calib["points_used"] >= 20
    assert calib["residual_in"]["mean"] < 2.0
    assert calib["residual_in"]["max"] < 3.0
    report = sheets.overlay(house, calib, calib_path.parent / calib["sheet"], tmp_path)
    ink = report["on_ink_by_kind"]
    # regression floors, not targets: today 94% / 86% / 72%
    assert ink["column"] >= 0.9
    assert ink["glass"] >= 0.8
    assert ink["wall"] >= 0.68
    assert (tmp_path / "foundation-plan.overlay.jpg").exists()
