"""Phase 1, step 2: flat roofs, the skylight, the flues, and the elevations.

The flues are the independent check: their plan positions come from the
roof plan, and the elevation fits (made only from the skylight domes and the
lettered levels) must put them where the elevations draw them.
"""

import importlib.util
import math

import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from architectworks.geom import mesh as g
from architectworks.resolve import resolve
from architectworks.resolve.roofs import roof_above
from architectworks.spec import load_spec
from architectworks.spec.model import BuildingSpec
from tests.conftest import ROOT

HOUSE = ROOT / "projects" / "example-house" / "house.yaml"
FT = 0.3048
IN = 0.0254


@pytest.fixture(scope="module")
def house_spec():
    return load_spec(HOUSE)


@pytest.fixture(scope="module")
def house(house_spec):
    return resolve(house_spec)


def _placements(house, element):
    return [p for p in house["placements"] if p["element"] == element]


def _world_verts(house, p):
    comp = next(c for c in house["components"] if c["name"] == p["component"])
    m = np.array(p["matrix"])
    v = np.array(comp["mesh"]["verts"])
    return (m[:3, :3] @ v.T).T + m[:3, 3]


def test_sphere_and_cylinder_are_closed():
    s = g.sphere(0.5, 24, 12)
    assert g.is_closed(s) and g.is_planar(s)
    assert g.volume(s) == pytest.approx(4 / 3 * math.pi * 0.125, rel=0.04)
    c = g.cylinder(0.15, 0.0, 2.0, 24)
    assert g.is_closed(c) and g.volume(c) == pytest.approx(math.pi * 0.15**2 * 2.0, rel=0.02)


def test_flat_roofs_cover_the_halls(house):
    roofs = house["derived"]["flat_roofs"]
    assert set(roofs) == {"roof_hall_n", "roof_hall_nw", "roof_hall_s"}
    top = (107 + 4 / 12 - 100) * FT
    for rid, info in roofs.items():
        assert info["parts"] == 1, rid
        assert info["top_m"] == pytest.approx(top, abs=1e-4)
        verts = _world_verts(house, _placements(house, rid)[0])
        assert verts[:, 2].max() == pytest.approx(top, abs=1e-4)
        assert verts[:, 2].min() == pytest.approx(top - 6 * IN, abs=1e-4)
    # the music room and the halls round it: about 400 sq ft
    assert 33.0 < roofs["roof_hall_s"]["area_m2"] < 42.0
    # the strip between the north pavilions: about 5 ft wide and 15 ft long, with its round end
    assert 5.0 < roofs["roof_hall_n"]["area_m2"] < 10.0


def test_flat_roofs_stop_where_pavilion_roofs_rise_above_them(house_spec, house):
    """No flat roof is left inside the part of a pavilion's roof that is
    higher than it: rebuild that region from the pavilion roofs alone."""
    from architectworks.resolve.model import Model, Placement
    from architectworks.geom.mesh import Mesh

    model = Model()
    comps = {c["name"]: c for c in house["components"]}
    for p in house["placements"]:
        if p["category"] == "Roof" and p["element"].startswith("pav_"):
            c = comps[p["component"]]
            if p["component"] not in model.components:
                from architectworks.resolve.model import Component

                model.components[p["component"]] = Component(p["component"], "Roof", Mesh([tuple(v) for v in c["mesh"]["verts"]], [tuple(f) for f in c["mesh"]["faces"]]), "M", [], True, {})
            model.placements.append(Placement(p["id"], p["component"], "Roof", p["matrix"], p["element"], "rule", "high", {}))
    top = (107 + 4 / 12 - 100) * FT
    above = roof_above(model, top, skip_element="")
    for rid in house["derived"]["flat_roofs"]:
        verts = _world_verts(house, _placements(house, rid)[0])
        upper = verts[np.isclose(verts[:, 2], top)]
        inside = [Point(x, y) for x, y in upper[:, :2] if above.buffer(-0.02).contains(Point(x, y))]
        assert not inside, f"{rid}: {len(inside)} vertices under a pavilion roof"


def test_skylight_stands_on_the_music_room_roof(house):
    glass = next(p for p in house["placements"] if p["id"] == "roof_hall_s.skylight0.glass")
    v = _world_verts(house, glass)
    top = (107 + 4 / 12 - 100) * FT
    assert v[:, 2].min() == pytest.approx(top, abs=1e-3)
    assert v[:, 2].max() == pytest.approx(top + (2 + 3 / 12) * FT, abs=0.03)
    music = np.array(house["points"]["cp_music"])
    centre = v[:, :2].mean(axis=0)
    assert np.hypot(*(centre - music)) < 4.0  # on the music room roof
    # the glass rises toward plan +Y, so the NW elevation sees its face
    hi = v[v[:, 2] > v[:, 2].max() - 0.05][:, 1].mean()
    lo = v[v[:, 2] < top + 0.05][:, 1].mean()
    assert hi > lo


@pytest.mark.parametrize("flue, top_ft", [("flue_n", 115 + 3 / 12), ("flue_s", 114 + 8 / 12)])
def test_flues_rise_from_the_cores_to_their_ball_caps(house, flue, top_ft):
    p = _placements(house, flue)[0]
    v = _world_verts(house, p)
    assert v[:, 2].max() == pytest.approx((top_ft - 100) * FT, abs=2e-3)
    assert v[:, 2].min() == pytest.approx((105 + 8 / 12 - 100) * FT, abs=1e-4)  # top of the core
    core = "core_n" if flue == "flue_n" else "core_s"
    centre = np.array(house["points"][flue])
    assert np.hypot(*(centre - np.array(house["points"][core]))) < (1 + 7 / 12 - 6 / 12) * FT  # pipe inside the core
    width = v[:, 0].max() - v[:, 0].min()
    assert width == pytest.approx((1 + 7 / 12) * FT, rel=0.01)  # the ball is the widest part


def test_new_stone_walls_close_the_music_room(house):
    for lid in ("link_living_music", "link_music_br"):
        (wall,) = _placements(house, lid)
        v = _world_verts(house, wall)
        assert v[:, 2].max() == pytest.approx((106 + 10 / 12 - 100) * FT, abs=1e-4)
        assert v[:, 2].min() == pytest.approx(-(1 + 8 / 12) * FT, abs=1e-4)  # from the music room floor


def test_flat_roof_needs_known_points(house_spec):
    data = house_spec.model_dump(mode="json")
    for el in data["elements"]:
        if el["id"] == "roof_hall_n":
            el["outline"][0] = "no_such_point"
    spec = BuildingSpec.model_validate(data)
    with pytest.raises(ValueError, match="unknown point 'no_such_point'"):
        resolve(spec)


def test_flue_type_must_be_a_flue(house_spec):
    data = house_spec.model_dump(mode="json")
    for el in data["elements"]:
        if el["id"] == "flue_n":
            el["type"] = "wall_stone"
    with pytest.raises(ValueError, match="is a wall, expected a flue"):
        BuildingSpec.model_validate(data)


# ---- elevations ----------------------------------------------------------------

needs_cv2 = pytest.mark.skipif(importlib.util.find_spec("cv2") is None, reason="needs the sheets extra (OpenCV)")


@pytest.fixture(scope="module")
def fits(house_spec, house):
    from architectworks import sheets
    from architectworks.units import to_metres

    datum = house_spec.levels[house_spec.datum].elev
    levels = {k: to_metres(lv.elev - datum) for k, lv in house_spec.levels.items()}
    return sheets.fit_elevations(ROOT / "projects" / "example-house" / "sheets" / "exterior-elevations.calib.yaml", house["points"], levels, float(datum) / 12)


@needs_cv2
def test_elevation_fits_are_consistent(fits):
    for name, v in fits["views"].items():
        across, up = v["px_per_ft"]
        assert 14.0 < across < 16.5 and 14.0 < up < 16.5, name
        assert abs(across - up) / up < 0.05, name  # one drawing scale, one photo
        worst = max(abs(e) for part in v["residual_px"].values() for e in part.values())
        assert worst < 6.0, (name, v["residual_px"])


@needs_cv2
@pytest.mark.parametrize("view, flue, drawn_x, drawn_top_y", [
    ("southwest", "flue_s", 586.5, 254), ("southwest", "flue_n", 993.2, 245),
    ("northwest", "flue_n", 733.2, 707), ("northwest", "flue_s", 768.0, 718),
])
def test_elevations_put_the_flues_where_they_are_drawn(fits, house, view, flue, drawn_x, drawn_top_y):
    """The fits know nothing about the flues; the flues' plan positions come
    from the roof plan and their heights from both elevations together."""
    v = fits["views"][view]
    x, y = house["points"][flue]
    h = x * v["h_axis"][0] + y * v["h_axis"][1]
    sx, bx = v["x"]
    px = sx * h + bx
    assert px == pytest.approx(drawn_x, abs=4.0)
    p = next(p for p in house["placements"] if p["element"] == flue)
    top = _world_verts(house, p)[:, 2].max()
    sy, by, tilt = v["y"]
    assert sy * top + by + tilt * px == pytest.approx(drawn_top_y, abs=5.0)


@needs_cv2
def test_compare_elevation_lays_the_drawing_over_a_render(fits, tmp_path):
    """A fake render: an opaque box covering one pavilion dome's position."""
    import cv2

    from architectworks import sheets

    v = fits["views"]["southwest"]
    ppm, h0, ztop = 60.0, -16.0, 6.0
    rgba = np.zeros((480, 1200, 4), np.uint8)
    rgba[200:470, 300:900] = (90, 90, 90, 255)
    (tmp_path / "preview").mkdir()
    cv2.imwrite(str(tmp_path / "preview" / "r.png"), rgba)
    render = {"looking": "-X", "file": "preview/r.png", "h0_m": h0, "z_top_m": ztop, "px_per_m": ppm, "size": [1200, 480]}
    drawing = sheets.load_elevation_sheet(ROOT / "projects" / "example-house" / "sheets" / "exterior-elevations.calib.yaml")
    res = sheets.compare_elevation(v, render, tmp_path, drawing, tmp_path / "c.jpg", "Southwest")
    img = cv2.imread(str(tmp_path / "c.jpg"))
    assert img.shape[1] == 1200 and img.shape[0] == 3 * (480 + 44)
    assert 0.0 < res["drawing_ink_inside_model"] < 1.0
