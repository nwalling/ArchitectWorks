"""Transforms, and the one place where Blender space becomes Unreal space.

Blender: metres, right-handed, Z up. Unreal: centimetres, left-handed, Z up,
X forward, Y right. The FBX importer maps a Blender point (x, y, z) to Unreal
(100x, -100y, 100z): a mirror across the XZ plane. For a placement that means

    location_ue = S * t * 100          S = diag(1, -1, 1)
    rotation_ue = S * R * S            quaternion (w, x, y, z) -> (w, -x, y, -z)

so that every vertex lands where it would if the whole scene were mirrored.
The phase 0 import test in Unreal checks this assumption against the real
importer, using the asymmetric marker.
"""

from __future__ import annotations

import math
from typing import Sequence

from architectworks.geom.mesh import Vec3

Matrix = tuple[tuple[float, float, float, float], ...]

IDENTITY: Matrix = ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))


def frame(origin: Vec3, x_axis: Vec3, y_axis: Vec3, z_axis: Vec3) -> Matrix:
    """A matrix whose columns are the local axes and whose translation is the origin."""
    return (
        (x_axis[0], y_axis[0], z_axis[0], origin[0]),
        (x_axis[1], y_axis[1], z_axis[1], origin[1]),
        (x_axis[2], y_axis[2], z_axis[2], origin[2]),
        (0.0, 0.0, 0.0, 1.0),
    )


def translation(t: Vec3) -> Matrix:
    return frame(t, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def rotation_z(deg: float, origin: Vec3 = (0.0, 0.0, 0.0)) -> Matrix:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return frame(origin, (c, s, 0.0), (-s, c, 0.0), (0.0, 0.0, 1.0))


def matmul(a: Matrix, b: Matrix) -> Matrix:
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)) for i in range(4))  # type: ignore[return-value]


def apply(m: Matrix, p: Vec3) -> Vec3:
    return (
        m[0][0] * p[0] + m[0][1] * p[1] + m[0][2] * p[2] + m[0][3],
        m[1][0] * p[0] + m[1][1] * p[1] + m[1][2] * p[2] + m[1][3],
        m[2][0] * p[0] + m[2][1] * p[1] + m[2][2] * p[2] + m[2][3],
    )


def rounded(m: Matrix, ndigits: int = 9) -> list[list[float]]:
    return [[round(v, ndigits) + 0.0 for v in row] for row in m]


def quat_from_matrix(m: Matrix) -> tuple[float, float, float, float]:
    """Unit quaternion (w, x, y, z) of the rotation part of ``m``."""
    r00, r01, r02 = m[0][0], m[0][1], m[0][2]
    r10, r11, r12 = m[1][0], m[1][1], m[1][2]
    r20, r21, r22 = m[2][0], m[2][1], m[2][2]
    tr = r00 + r11 + r22
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (r21 - r12) / s, (r02 - r20) / s, (r10 - r01) / s
    elif r00 > r11 and r00 > r22:
        s = math.sqrt(1.0 + r00 - r11 - r22) * 2
        w, x, y, z = (r21 - r12) / s, 0.25 * s, (r01 + r10) / s, (r02 + r20) / s
    elif r11 > r22:
        s = math.sqrt(1.0 + r11 - r00 - r22) * 2
        w, x, y, z = (r02 - r20) / s, (r01 + r10) / s, 0.25 * s, (r12 + r21) / s
    else:
        s = math.sqrt(1.0 + r22 - r00 - r11) * 2
        w, x, y, z = (r10 - r01) / s, (r02 + r20) / s, (r12 + r21) / s, 0.25 * s
    if w < 0:
        w, x, y, z = -w, -x, -y, -z
    n = math.sqrt(w * w + x * x + y * y + z * z)
    return (w / n, x / n, y / n, z / n)


def matrix_from_quat(q: Sequence[float]) -> list[list[float]]:
    w, x, y, z = q
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]


# ---- Blender -> Unreal ------------------------------------------------------

CM_PER_M = 100.0


def ue_point_cm(p: Vec3) -> Vec3:
    return (p[0] * CM_PER_M, -p[1] * CM_PER_M, p[2] * CM_PER_M)


def ue_quat_xyzw(m: Matrix) -> tuple[float, float, float, float]:
    """Unreal FQuat components (x, y, z, w) for a Blender placement matrix."""
    w, x, y, z = quat_from_matrix(m)
    return (-x, y, -z, w)


def ue_rotator_from_quat(q_xyzw: Sequence[float]) -> dict[str, float]:
    """FQuat::Rotator(), transcribed. Informative only: the import script hands
    Unreal the quaternion and lets Unreal do this conversion itself."""
    x, y, z, w = q_xyzw
    singularity = z * x - w * y
    yaw_y = 2.0 * (w * z + x * y)
    yaw_x = 1.0 - 2.0 * (y * y + z * z)
    threshold = 0.4999995
    deg = 180.0 / math.pi
    if singularity < -threshold:
        pitch = -90.0
        yaw = math.atan2(yaw_y, yaw_x) * deg
        roll = _normalize_axis(-yaw - 2.0 * math.atan2(x, w) * deg)
    elif singularity > threshold:
        pitch = 90.0
        yaw = math.atan2(yaw_y, yaw_x) * deg
        roll = _normalize_axis(yaw - 2.0 * math.atan2(x, w) * deg)
    else:
        pitch = math.asin(2.0 * singularity) * deg
        yaw = math.atan2(yaw_y, yaw_x) * deg
        roll = math.atan2(-2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)) * deg
    return {"roll": roll, "pitch": pitch, "yaw": yaw}


def ue_quat_from_rotator(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """FRotator::Quaternion(), transcribed; used by tests to check the round trip."""
    r = math.pi / 360.0
    sp, cp = math.sin(pitch * r), math.cos(pitch * r)
    sy, cy = math.sin(yaw * r), math.cos(yaw * r)
    sr, cr = math.sin(roll * r), math.cos(roll * r)
    return (
        cr * sp * sy - sr * cp * cy,
        -cr * sp * cy - sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def _normalize_axis(a: float) -> float:
    a = math.fmod(a, 360.0)
    if a > 180.0:
        a -= 360.0
    elif a < -180.0:
        a += 360.0
    return a


def transformed_box_bounds(m: Matrix, bmin: Vec3, bmax: Vec3) -> tuple[Vec3, Vec3]:
    """World AABB of a local box under ``m``, computed the way Unreal computes
    component bounds (transform the local box, then take its AABB)."""
    corners = [(x, y, z) for x in (bmin[0], bmax[0]) for y in (bmin[1], bmax[1]) for z in (bmin[2], bmax[2])]
    pts = [apply(m, c) for c in corners]
    xs, ys, zs = zip(*pts)
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def ue_bounds_cm(bmin: Vec3, bmax: Vec3) -> tuple[Vec3, Vec3]:
    """A Blender-space AABB (m) as an Unreal-space AABB (cm)."""
    a, b = ue_point_cm(bmin), ue_point_cm(bmax)
    return (min(a[0], b[0]), min(a[1], b[1]), min(a[2], b[2])), (max(a[0], b[0]), max(a[1], b[1]), max(a[2], b[2]))
