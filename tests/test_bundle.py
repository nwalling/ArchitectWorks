import pytest

from architectworks.unreal_bundle import build_manifest


@pytest.fixture(scope="module")
def manifest(resolved):
    return build_manifest(resolved)


def test_marker_local_bounds_are_mirrored_and_in_cm(manifest):
    m = next(x for x in manifest["meshes"] if "Marker" in x["name"])
    assert m["local_bounds_cm"]["min"] == [0.0, -100.0, 0.0]
    assert m["local_bounds_cm"]["max"] == [200.0, 0.0, 150.0]
    assert m["collision"]["convex_hulls"] == 3


def test_marker_placement(manifest):
    p = next(x for x in manifest["placements"] if x["id"] == "aw.handedness_marker")
    assert p["location_cm"] == [-600.0, -900.0, 0.0]
    assert p["rotation_rotator_deg"]["yaw"] == pytest.approx(-30.0)


def test_every_placement_has_a_mesh(manifest):
    names = {m["name"] for m in manifest["meshes"]}
    assert all(p["mesh"] in names for p in manifest["placements"])
    assert all(m["file"].endswith(".fbx") for m in manifest["meshes"])


def test_building_bounds(manifest):
    b = manifest["expected_building_bounds_cm"]
    width = b["max"][0] - b["min"][0]
    assert width == pytest.approx(2 * (12 * 12 + 8) * 2.54, abs=0.05)
    assert b["min"][2] == pytest.approx(-4 * 2.54, abs=0.01)  # bottom of the 4" slab


def test_only_conflicts_travel_with_the_manifest(manifest):
    assert {c["status"] for c in manifest["checks"]} == {"conflict"}
