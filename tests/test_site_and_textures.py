"""Textures and the site: sun, north, texture coordinates, asset fetching,
terrain and planting rules."""

import hashlib
import importlib.util
import json
import math
from collections import defaultdict

import numpy as np
import pytest

from architectworks import assets
from architectworks.resolve import resolve
from architectworks.site import plan_direction, sun_position
from architectworks.spec import load_spec
from tests.conftest import ROOT

HOUSE = ROOT / "projects" / "example-house" / "house.yaml"


@pytest.fixture(scope="module")
def spec():
    return load_spec(HOUSE)


@pytest.fixture(scope="module")
def house(spec):
    return resolve(spec)


def test_sun_at_solar_noon_on_the_solstice():
    """At 38.5 N the noon sun on the June solstice stands 90 - 38.5 + 23.44
    degrees up, due south; at 122.7 W on Pacific daylight time that is about
    13:12 by the clock."""
    best = max(((sun_position(38.5, -122.7, -7, "2026-06-21", f"13:{m:02d}"), m) for m in range(0, 30)), key=lambda t: t[0][1])
    (az, el), minute = best
    assert el == pytest.approx(90 - 38.5 + 23.44, abs=0.3)
    assert az == pytest.approx(180, abs=3)
    assert 5 <= minute <= 20


def test_afternoon_sun_is_in_the_south_west():
    az, el = sun_position(38.5, -122.7, -7, "2026-09-20", "16:30")
    assert 235 < az < 255 and 25 < el < 35


def test_true_north_from_the_drawings(house):
    # the roof plan's arrow points south toward the sheet's upper right
    north = house["site"]["north"]
    assert north == pytest.approx([-math.sqrt(0.5), -math.sqrt(0.5)], abs=1e-9)
    # and the elevations' names agree: south-west is plan +X, north-west plan -Y
    assert plan_direction(225.0, 225.0) == pytest.approx((1.0, 0.0), abs=1e-9)
    assert plan_direction(225.0, 315.0) == pytest.approx((0.0, -1.0), abs=1e-9)


def test_site_in_resolved_output(house):
    site = house["site"]
    assert site["grade_m"] == pytest.approx(-(5 / 12) * 0.3048, abs=1e-6)  # 99'-7"
    sun = site["sun"]["direction"]
    assert sun[0] > 0.7 and sun[2] > 0.4  # late afternoon: over the glazed south-west side
    assert len(site["planting"]) >= 10 and site["outline"]


def test_shingle_courses_run_level(house):
    """On a cone roof, every vertex at one height has the same v, so the
    shingle courses are level."""
    comp = next(c for c in house["components"] if c["name"].endswith("_PavTl_RoofConical"))
    assert comp["mesh"]["uv_mode"] == "cone"
    verts, faces, uvs = comp["mesh"]["verts"], comp["mesh"]["faces"], comp["mesh"]["uv"]
    by_z = defaultdict(set)
    for f, fuv in zip(faces, uvs):
        pts = [verts[i] for i in f]
        n = np.cross(np.subtract(pts[1], pts[0]), np.subtract(pts[2], pts[0]))
        n = n / np.linalg.norm(n)
        if 0.25 < abs(n[2]) < 0.98:
            for (x, y, z), (u, v) in zip(pts, fuv):
                by_z[(n[2] > 0, round(z, 4))].add(v)  # top and underside apart
    assert by_z and all(max(vs) - min(vs) <= 0.005 for vs in by_z.values())  # within 5 mm


def test_curved_walls_wrap(house):
    comp = next(c for c in house["components"] if c["name"].endswith("_PavTl_WallStoneArc"))
    assert comp["mesh"]["uv_mode"] == "cylinder"
    verts, faces = comp["mesh"]["verts"], comp["mesh"]["faces"]
    us = []
    for f, fuv in zip(faces, comp["mesh"]["uv"]):
        pts = [verts[i] for i in f]
        n = np.cross(np.subtract(pts[1], pts[0]), np.subtract(pts[2], pts[0]))
        if abs(n[2]) / np.linalg.norm(n) < 0.7:  # the wall faces, not the top and bottom
            us += [u for u, v in fuv]
    # the arc's outside is 11'-8" plus 5" of facing over 238..303 degrees, plus its short returns
    arc = math.radians(65) * (11 + 13 / 12) * 0.3048
    assert arc <= max(us) - min(us) <= arc * 1.15


def test_spec_names_textures_and_plants(spec):
    want = assets.wanted(spec)
    ids = {t for t, _ in want["textures"]}
    assert {"stone_wall", "roof_slates_02", "withered_grass", "rough_wood"} <= ids
    assert "island_tree_01" in want["models"] and "fir_tree_01" in want["models"]


def test_download_checks_the_md5(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"x" * 1000)
    good = hashlib.md5(src.read_bytes()).hexdigest()
    dest = tmp_path / "out" / "a.bin"
    assert assets._download(src.as_uri(), good, dest, lambda *_: None) == 1000
    assert assets._download(src.as_uri(), good, dest, lambda *_: None) == 0  # already there
    with pytest.raises(IOError, match="md5"):
        assets._download(src.as_uri(), "0" * 32, tmp_path / "out" / "b.bin", lambda *_: None)
    assert not (tmp_path / "out" / "b.bin").exists()


def test_resolved_assets_reads_the_index(spec, tmp_path):
    (tmp_path / "textures" / "stone_wall").mkdir(parents=True)
    for m in ("diff", "nor", "rough"):
        (tmp_path / "textures" / "stone_wall" / f"stone_wall_{m}_2k.jpg").write_bytes(b"jpg")
    idx = {"textures": {"stone_wall": {"dimensions_mm": [2000, 2000], "resolution": "2k", "maps": {m: f"textures/stone_wall/stone_wall_{m}_2k.jpg" for m in ("diff", "nor", "rough")}}}, "models": {}}
    (tmp_path / "index.json").write_text(json.dumps(idx))
    found = assets.resolved_assets(spec, tmp_path)
    assert list(found["materials"]) == ["M_StoneNative"]
    assert found["materials"]["M_StoneNative"]["tile_m"] == pytest.approx(2.0)
    assert assets.resolved_assets(spec, tmp_path / "nothing") is None


@pytest.fixture(scope="module")
def landscape():
    if importlib.util.find_spec("bpy") is None:
        pytest.skip("needs bpy")
    spec_ = importlib.util.spec_from_file_location("aw_landscape_test", ROOT / "blendgen" / "landscape.py")
    mod = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(mod)
    return mod


def test_terrain_pad_and_planting_bands(house, landscape):
    site = house["site"]
    T = landscape.Terrain(site)
    X, Y = np.meshgrid(np.linspace(T.cx - 25, T.cx + 25, 120), np.linspace(T.cy - 25, T.cy + 25, 120))
    X, Y = X.ravel(), Y.ravel()
    d = landscape._seg_distance(X, Y, site["outline"])
    Z = T.height(X, Y, d)
    pad = site["terrain"]["pad_m"]
    on_pad = (d > 0.2) & (d < pad - 0.2)
    assert on_pad.sum() > 100 and np.allclose(Z[on_pad], site["grade_m"])
    # under the building the ground drops below the sunken music room's slab,
    # but not until 0.3 m inside the footprints (the outline is 0.3 m out)
    under = d < -T.UNDER
    assert under.sum() > 100 and Z[under].max() < -0.61
    assert np.allclose(Z[(d > -T.UNDER + 0.05) & (d < 0.2)], site["grade_m"])
    # and the grid is fine enough there that the slope between stays inside
    xs, _ = T.grid()
    assert np.diff(xs[np.abs(xs - T.cx) < T.R0]).max() <= T.UNDER - 0.3 + 1e-9
    # beyond the pad it eases into the terrain
    assert np.abs(Z[d > pad + site["terrain"]["blend_m"]] - site["grade_m"]).max() > 0.05
    # scattered plants keep to their band and their spacing
    plant = {"count": 12, "near_m": 10.0, "far_m": 60.0, "spacing_m": 12.0}
    pts = landscape.scatter(T, plant, 3)
    d = landscape._seg_distance(pts[:, 0], pts[:, 1], site["outline"])
    assert len(pts) == 12 and d.min() >= 10.0 and d.max() <= 60.0
    gaps = np.hypot(*(pts[:, None, :] - pts[None, :, :]).transpose(2, 0, 1)) + np.eye(len(pts)) * 1e9
    assert gaps.min() >= 12.0


def test_camera_foregrounds_stay_clear(house, landscape):
    """Trees, shrubs and rocks keep out of the wedge between an eye-level
    camera and the building; grass (no spacing) does not."""
    site = house["site"]
    T = landscape.Terrain(site)
    cam, tgt = (T.cx + 40.0, T.cy), (T.cx, T.cy)
    views = [(cam, tgt, 12.0)]
    shrub = {"count": 30, "near_m": 3.0, "far_m": 40.0, "spacing_m": 2.0}
    pts = landscape.scatter(T, shrub, 5, views)
    assert len(pts) == 30 and not landscape.in_view(pts[:, 0], pts[:, 1], views).any()
    # the wedge is narrow at the camera and wide at the building
    assert landscape.in_view(np.array([T.cx + 38.0, T.cx + 5.0]), np.array([T.cy + 1.0, T.cy + 12.0]), views).all()
    assert not landscape.in_view(np.array([T.cx + 38.0, T.cx + 50.0]), np.array([T.cy + 4.0, T.cy]), views).any()
    grass = {"count": 400, "near_m": 3.0, "far_m": 40.0, "spacing_m": 0.0}
    g = landscape.scatter(T, grass, 5, views)
    assert landscape.in_view(g[:, 0], g[:, 1], views).any()


def test_ground_cover_thins_toward_its_far_edge(house, landscape):
    T = landscape.Terrain(house["site"])
    grass = {"count": 3000, "near_m": 2.0, "far_m": 40.0, "spacing_m": 0.0}
    g = landscape.scatter(T, grass, 9)
    d = landscape._seg_distance(g[:, 0], g[:, 1], house["site"]["outline"])
    inner, outer = ((d >= 2) & (d < 21)).sum(), ((d >= 30) & (d <= 40)).sum()
    assert inner > 1000 and outer < inner * 0.25


def test_texture_search_noise_is_counted_not_kept():
    from architectworks.cli import quiet_texture_search

    text = "a\nFailed to open dir (x): /y\nb\nFailed to open dir (x): /z"
    assert quiet_texture_search(text).splitlines() == ["a", "b", "(2 'Failed to open dir' lines from the FBX importer's texture search left out)"]
    assert quiet_texture_search("a\nb") == "a\nb"
