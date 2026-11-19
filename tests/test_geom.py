import math
import random

import pytest

from architectworks.geom import mesh as g
from architectworks.geom import xform as x


def check_solid(m: g.Mesh) -> None:
    assert g.is_closed(m)
    assert g.volume(m) > 0
    assert g.is_planar(m)


def test_box_volume_and_orientation():
    m = g.box(0, 2, 0, 3, 0, 4)
    check_solid(m)
    assert g.volume(m) == pytest.approx(24.0)
    assert g.is_convex(m)


def test_prism_accepts_either_winding():
    sq = [(0, 0), (1, 0), (1, 1), (0, 1)]
    a, b = g.prism(sq, 0, 1), g.prism(list(reversed(sq)), 0, 1)
    assert g.volume(a) == pytest.approx(1.0) == pytest.approx(g.volume(b))


def test_concave_prism_is_closed_with_correct_volume():
    # an L-shaped footprint, area 3
    poly = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
    m = g.prism(poly, 0, 2)
    check_solid(m)
    assert g.volume(m) == pytest.approx(6.0)
    assert not g.is_convex(m)


def test_annular_sector_volume_approaches_analytic():
    r_in, r_out, h = 3.0, 3.5, 2.0
    m = g.annular_sector(r_in, r_out, 180, 360, 96, 0, h)
    check_solid(m)
    exact = 0.5 * math.pi * (r_out**2 - r_in**2) * h
    assert g.volume(m) == pytest.approx(exact, rel=1e-3)


def test_single_segment_sector_is_convex():
    assert g.is_convex(g.annular_sector(3.0, 3.5, 0, 15, 1, 0, 2))


def test_ring_is_closed():
    m = g.ring(1.0, 1.2, 0, 0.2, 48)
    check_solid(m)


def test_dome_volume_approaches_spherical_cap():
    r, h = 0.8, 0.3
    m = g.dome(r, h, segments=128, rings=24)
    assert g.is_closed(m) and g.volume(m) > 0
    exact = math.pi * h * (3 * r * r + h * h) / 6
    assert g.volume(m) == pytest.approx(exact, rel=5e-3)


def test_plate_follows_its_plane():
    poly = [(0, 0, 0), (2, 0, 0), (2, 1, 1), (0, 1, 1)]  # sloped 45 degrees
    m = g.plate(poly, 0.1)
    check_solid(m)
    assert g.volume(m) == pytest.approx(2 * math.sqrt(2) * 0.1)


def test_box_uvs_one_per_corner():
    m = g.box(0, 1, 0, 1, 0, 1)
    uvs = g.box_uvs(m)
    assert [len(u) for u in uvs] == [len(f) for f in m.faces]


def test_ray_exit_from_column_profile():
    # D-1 column: the glazing line leaves the column 5.80 in from its C.P.
    inch = 0.0254
    prof = [(2, 8), (-2, 8), (-6.75, 0), (-2, -8), (2, -8), (6.75, 0)]
    prof = [(a * inch, b * inch) for a, b in prof]
    d = (-math.cos(math.radians(30)), -0.5)
    assert g.ray_exit(prof, d) / inch == pytest.approx(5.80, abs=0.01)


# ---- Blender -> Unreal ------------------------------------------------------------

S = ((1, 0, 0), (0, -1, 0), (0, 0, 1))


def random_rotation(rng: random.Random) -> x.Matrix:
    m = x.rotation_z(rng.uniform(-180, 180))
    a = math.radians(rng.uniform(-80, 80))
    rx = ((1, 0, 0, 0), (0, math.cos(a), -math.sin(a), 0), (0, math.sin(a), math.cos(a), 0), (0, 0, 0, 1))
    m = x.matmul(m, rx)
    return x.matmul(x.translation((rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-1, 3))), x.matmul(m, x.rotation_z(rng.uniform(-180, 180))))


def test_unreal_quaternion_is_mirrored_rotation():
    rng = random.Random(7)
    for _ in range(50):
        m = random_rotation(rng)
        qx, qy, qz, qw = x.ue_quat_xyzw(m)
        r_ue = x.matrix_from_quat((qw, qx, qy, qz))
        r = [row[:3] for row in m[:3]]
        srs = [[sum(S[i][k] * r[k][l] * S[l][j] for k in range(3) for l in range(3)) for j in range(3)] for i in range(3)]
        for i in range(3):
            for j in range(3):
                assert r_ue[i][j] == pytest.approx(srs[i][j], abs=1e-9)


def test_every_vertex_lands_mirrored():
    """Mesh imported mirrored + placement converted = whole scene mirrored."""
    rng = random.Random(11)
    for _ in range(20):
        m = random_rotation(rng)
        qx, qy, qz, qw = x.ue_quat_xyzw(m)
        r_ue = x.matrix_from_quat((qw, qx, qy, qz))
        t_ue = x.ue_point_cm((m[0][3], m[1][3], m[2][3]))
        for _ in range(5):
            v = (rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(-2, 2))
            v_ue = x.ue_point_cm(v)  # what the FBX importer does to a vertex
            world_ue = [sum(r_ue[i][j] * v_ue[j] for j in range(3)) + t_ue[i] for i in range(3)]
            expected = x.ue_point_cm(x.apply(m, v))
            assert world_ue == pytest.approx(list(expected), abs=1e-6)


def test_yaw_sign():
    # +90 degrees about Z in Blender (X toward Y) is yaw -90 in Unreal (X toward -Y).
    rot = x.ue_rotator_from_quat(x.ue_quat_xyzw(x.rotation_z(90)))
    assert rot["yaw"] == pytest.approx(-90.0)
    assert rot["pitch"] == pytest.approx(0.0, abs=1e-9)
    assert rot["roll"] == pytest.approx(0.0, abs=1e-9)


def test_rotator_round_trip():
    rng = random.Random(3)
    for _ in range(50):
        roll, pitch, yaw = rng.uniform(-179, 179), rng.uniform(-85, 85), rng.uniform(-179, 179)
        q = x.ue_quat_from_rotator(roll, pitch, yaw)
        r = x.ue_rotator_from_quat(q)
        assert (r["roll"], r["pitch"], r["yaw"]) == pytest.approx((roll, pitch, yaw), abs=1e-6)


def test_transformed_box_bounds_rotated():
    m = x.rotation_z(90)
    bmin, bmax = x.transformed_box_bounds(m, (0, 0, 0), (2, 1, 1))
    assert bmin == pytest.approx((-1, 0, 0))
    assert bmax == pytest.approx((0, 2, 1))


def test_extrude_polygon_with_a_hole():
    from shapely.geometry import Point, Polygon

    from architectworks.resolve.floors import extrude

    outer = Polygon([(0, 0), (6, 0), (6, 4), (0, 4)])
    shape = outer.difference(Point(3, 2).buffer(1, quad_segs=16))
    m = extrude(shape, -0.1, 0.0)
    assert g.is_closed(m)
    assert g.volume(m) == pytest.approx(shape.area * 0.1, rel=1e-6)


def test_glazed_run_may_wrap():
    from architectworks.spec.model import contiguous_run

    assert contiguous_run([5, 0, 1], 6) == (5, 3)
    assert contiguous_run([0, 1, 2], 6) == (0, 3)
    assert contiguous_run([], 6) == (0, 0)
    with pytest.raises(ValueError):
        contiguous_run([0, 2], 6)


def test_arc_spans_normalise():
    from architectworks.resolve.pavilion import in_span, norm_span

    assert norm_span(238, 303) == (238, 303)
    assert norm_span(330, 30) == (330, 390)
    assert norm_span(-90, 90) == (270, 450)
    assert in_span(10, 330, 390) and not in_span(100, 330, 390)
