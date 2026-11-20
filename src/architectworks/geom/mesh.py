"""Plain-Python meshes: built, checked and UV-mapped outside Blender.

Coordinates are metres in a right-handed, Z-up frame (Blender's). Faces are
vertex-index tuples wound counter-clockwise seen from outside, so the
right-hand rule gives the outward normal. Builders fix winding by checking the
sign of the enclosed volume, so each one only has to be internally consistent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

Vec3 = tuple[float, float, float]
Vec2 = tuple[float, float]


@dataclass
class Mesh:
    verts: list[Vec3] = field(default_factory=list)
    faces: list[tuple[int, ...]] = field(default_factory=list)
    face_mat: list[int] = field(default_factory=list)

    # -- construction -------------------------------------------------------
    def add(self, other: "Mesh") -> "Mesh":
        base = len(self.verts)
        self.verts.extend(other.verts)
        self.faces.extend(tuple(i + base for i in f) for f in other.faces)
        self.face_mat.extend(other.face_mat or [0] * len(other.faces))
        return self

    def flipped(self) -> "Mesh":
        return Mesh(list(self.verts), [tuple(reversed(f)) for f in self.faces], list(self.face_mat))

    def oriented_outward(self) -> "Mesh":
        return self.flipped() if volume(self) < 0 else self

    def transformed(self, fn: Callable[[Vec3], Vec3]) -> "Mesh":
        return Mesh([fn(v) for v in self.verts], list(self.faces), list(self.face_mat))

    def to_json(self, ndigits: int = 6) -> dict:
        return {
            "verts": [[round(c, ndigits) + 0.0 for c in v] for v in self.verts],
            "faces": [list(f) for f in self.faces],
            "face_mat": list(self.face_mat or [0] * len(self.faces)),
        }

    @property
    def triangle_count(self) -> int:
        return sum(len(f) - 2 for f in self.faces)


# ---- vector helpers ---------------------------------------------------------


def sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def scale(a: Vec3, s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Vec3, b: Vec3) -> Vec3:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def norm(a: Vec3) -> float:
    return math.sqrt(dot(a, a))


def unit(a: Vec3) -> Vec3:
    n = norm(a)
    if n < 1e-12:
        raise ValueError("zero-length vector")
    return (a[0] / n, a[1] / n, a[2] / n)


def polar(r: float, deg: float, z: float = 0.0) -> Vec3:
    t = math.radians(deg)
    return (r * math.cos(t), r * math.sin(t), z)


# ---- measurement ------------------------------------------------------------


def face_normal(mesh: Mesh, face: Sequence[int]) -> Vec3:
    """Newell's method: robust for planar n-gons, length = 2 x area."""
    nx = ny = nz = 0.0
    pts = [mesh.verts[i] for i in face]
    for i, p in enumerate(pts):
        q = pts[(i + 1) % len(pts)]
        nx += (p[1] - q[1]) * (p[2] + q[2])
        ny += (p[2] - q[2]) * (p[0] + q[0])
        nz += (p[0] - q[0]) * (p[1] + q[1])
    return (nx, ny, nz)


def volume(mesh: Mesh) -> float:
    """Signed enclosed volume (divergence theorem, fan per face)."""
    total = 0.0
    for f in mesh.faces:
        a = mesh.verts[f[0]]
        for i in range(1, len(f) - 1):
            b, c = mesh.verts[f[i]], mesh.verts[f[i + 1]]
            total += dot(a, cross(b, c))
    return total / 6.0


def bounds(mesh: Mesh) -> tuple[Vec3, Vec3]:
    xs, ys, zs = zip(*mesh.verts)
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def is_closed(mesh: Mesh) -> bool:
    """Every directed edge is matched by its reverse exactly once."""
    edges: dict[tuple[int, int], int] = {}
    for f in mesh.faces:
        for i in range(len(f)):
            e = (f[i], f[(i + 1) % len(f)])
            edges[e] = edges.get(e, 0) + 1
    return all(n == 1 and edges.get((b, a)) == 1 for (a, b), n in edges.items())


def is_planar(mesh: Mesh, tol: float = 1e-6) -> bool:
    for f in mesh.faces:
        if len(f) <= 3:
            continue
        n = unit(face_normal(mesh, f))
        p0 = mesh.verts[f[0]]
        if any(abs(dot(sub(mesh.verts[i], p0), n)) > tol for i in f):
            return False
    return True


def is_convex(mesh: Mesh, tol: float = 1e-6) -> bool:
    """True if every vertex lies on or behind every face plane."""
    for f in mesh.faces:
        n = face_normal(mesh, f)
        if norm(n) < 1e-12:
            return False
        n = unit(n)
        p0 = mesh.verts[f[0]]
        if any(dot(sub(v, p0), n) > tol for v in mesh.verts):
            return False
    return True


# ---- UVs ---------------------------------------------------------------------


def box_uvs(mesh: Mesh, metres_per_tile: float = 1.0) -> list[list[Vec2]]:
    """Box projection at a fixed real-world density: one UV unit per tile."""
    out: list[list[Vec2]] = []
    s = 1.0 / metres_per_tile
    for f in mesh.faces:
        nx, ny, nz = (abs(c) for c in face_normal(mesh, f))
        uvs = []
        for i in f:
            x, y, z = mesh.verts[i]
            if nz >= nx and nz >= ny:
                uvs.append((x * s, y * s))
            elif nx >= ny:
                uvs.append((y * s, z * s))
            else:
                uvs.append((x * s, z * s))
        out.append(uvs)
    return out


# ---- builders ----------------------------------------------------------------


def polygon_area(poly: Sequence[Vec2]) -> float:
    return 0.5 * sum(poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1] for i in range(len(poly)))


def prism(poly: Sequence[Vec2], z0: float | Sequence[float], z1: float | Sequence[float], mat: int = 0) -> Mesh:
    """Extrude a simple 2D polygon between bottom and top heights.

    ``z0``/``z1`` may be per-vertex sequences for sloped tops or bottoms.
    """
    n = len(poly)
    if polygon_area(poly) < 0:
        poly = list(reversed(poly))
        if not isinstance(z0, (int, float)):
            z0 = list(reversed(z0))
        if not isinstance(z1, (int, float)):
            z1 = list(reversed(z1))
    zb = [z0] * n if isinstance(z0, (int, float)) else list(z0)
    zt = [z1] * n if isinstance(z1, (int, float)) else list(z1)
    verts = [(p[0], p[1], zb[i]) for i, p in enumerate(poly)] + [(p[0], p[1], zt[i]) for i, p in enumerate(poly)]
    faces: list[tuple[int, ...]] = [tuple(reversed(range(n))), tuple(range(n, 2 * n))]
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, n + j, n + i))
    return Mesh(verts, faces, [mat] * len(faces)).oriented_outward()


def box(x0: float, x1: float, y0: float, y1: float, z0: float, z1: float, mat: int = 0) -> Mesh:
    return prism([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], z0, z1, mat)


def loft(sections: Sequence[Sequence[Vec3]], closed: bool = False, mat: int = 0) -> Mesh:
    """Connect cross-sections (same vertex count, same order) with quads.

    Open lofts are capped at both ends; ``closed`` joins the last section
    back to the first (a ring).
    """
    k = len(sections[0])
    if any(len(s) != k for s in sections):
        raise ValueError("all sections need the same number of points")
    verts = [p for s in sections for p in s]
    faces: list[tuple[int, ...]] = []
    m = len(sections)
    spans = m if closed else m - 1
    for i in range(spans):
        a, b = i * k, ((i + 1) % m) * k
        for j in range(k):
            jn = (j + 1) % k
            faces.append((a + j, a + jn, b + jn, b + j))
    if not closed:
        faces.append(tuple(reversed(range(k))))
        faces.append(tuple(range((m - 1) * k, m * k)))
    return Mesh(verts, faces, [mat] * len(faces)).oriented_outward()


def radial_section(deg: float, pts_rz: Sequence[Vec2]) -> list[Vec3]:
    """A cross-section given as (radius, z) pairs, swung to angle ``deg``."""
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return [(r * c, r * s, z) for r, z in pts_rz]


def angles(a0: float, a1: float, segments: int) -> list[float]:
    return [a0 + (a1 - a0) * i / segments for i in range(segments + 1)]


def annular_sector(r_in: float, r_out: float, a0: float, a1: float, segments: int, z0: float, z1: float, mat: int = 0) -> Mesh:
    rect = [(r_in, z0), (r_out, z0), (r_out, z1), (r_in, z1)]
    return loft([radial_section(a, rect) for a in angles(a0, a1, segments)], mat=mat)


def ring(r_in: float, r_out: float, z0: float, z1: float, segments: int, mat: int = 0) -> Mesh:
    rect = [(r_in, z0), (r_out, z0), (r_out, z1), (r_in, z1)]
    return loft([radial_section(360.0 * i / segments, rect) for i in range(segments)], closed=True, mat=mat)


def plate(poly: Sequence[Vec3], thickness: float, up: Vec3 = (0.0, 0.0, 1.0), mat: int = 0) -> Mesh:
    """A planar polygon given on its lower face, thickened along its normal
    (the normal is chosen to point toward ``up``)."""
    tmp = Mesh(list(poly), [tuple(range(len(poly)))])
    n = unit(face_normal(tmp, tmp.faces[0]))
    if dot(n, up) < 0:
        n = scale(n, -1.0)
    off = scale(n, thickness)
    bottom = list(poly)
    top = [add(p, off) for p in poly]
    k = len(poly)
    verts = bottom + top
    faces: list[tuple[int, ...]] = [tuple(reversed(range(k))), tuple(range(k, 2 * k))]
    for i in range(k):
        j = (i + 1) % k
        faces.append((i, j, k + j, k + i))
    return Mesh(verts, faces, [mat] * len(faces)).oriented_outward()


def dome(radius: float, rise: float, segments: int = 32, rings: int = 6, mat: int = 0) -> Mesh:
    """A spherical cap standing on z=0, closed by a flat base."""
    rs = (radius * radius + rise * rise) / (2.0 * rise)
    cz = rise - rs
    tmax = math.asin(min(1.0, radius / rs))
    verts: list[Vec3] = [(0.0, 0.0, rise)]
    for i in range(1, rings + 1):
        t = tmax * i / rings
        r, z = rs * math.sin(t), cz + rs * math.cos(t)
        if i == rings:
            r, z = radius, 0.0
        for j in range(segments):
            a = 2 * math.pi * j / segments
            verts.append((r * math.cos(a), r * math.sin(a), z))
    faces: list[tuple[int, ...]] = []
    for j in range(segments):
        faces.append((0, 1 + j, 1 + (j + 1) % segments))
    for i in range(rings - 1):
        a, b = 1 + i * segments, 1 + (i + 1) * segments
        for j in range(segments):
            jn = (j + 1) % segments
            faces.append((a + j, b + j, b + jn, a + jn))
    last = 1 + (rings - 1) * segments
    faces.append(tuple(last + j for j in reversed(range(segments))))
    return Mesh(verts, faces, [mat] * len(faces)).oriented_outward()


def sphere(radius: float, segments: int = 24, rings: int = 12, mat: int = 0) -> Mesh:
    """A closed UV sphere centred on the origin."""
    verts: list[Vec3] = [(0.0, 0.0, radius)]
    for i in range(1, rings):
        t = math.pi * i / rings
        for j in range(segments):
            a = 2 * math.pi * j / segments
            verts.append((radius * math.sin(t) * math.cos(a), radius * math.sin(t) * math.sin(a), radius * math.cos(t)))
    bottom = len(verts)
    verts.append((0.0, 0.0, -radius))
    faces: list[tuple[int, ...]] = []
    for j in range(segments):
        faces.append((0, 1 + j, 1 + (j + 1) % segments))
    for i in range(rings - 2):
        a, b = 1 + i * segments, 1 + (i + 1) * segments
        for j in range(segments):
            jn = (j + 1) % segments
            faces.append((a + j, b + j, b + jn, a + jn))
    last = 1 + (rings - 2) * segments
    for j in range(segments):
        faces.append((last + j, bottom, last + (j + 1) % segments))
    return Mesh(verts, faces, [mat] * len(faces)).oriented_outward()


def cylinder(radius: float, z0: float, z1: float, segments: int = 24, mat: int = 0) -> Mesh:
    poly = [(radius * math.cos(2 * math.pi * i / segments), radius * math.sin(2 * math.pi * i / segments)) for i in range(segments)]
    return prism(poly, z0, z1, mat)


def glazing_unit(length: float, height: float, frame_w: float, frame_d: float, glass_t: float, panels: int = 2) -> tuple[Mesh, Mesh]:
    """A framed glazing unit in its own frame: x along the opening (0 to
    length), y across it (the glazing line at 0), z up (0 to height).
    Returns (frame, glass). Two panels are a sliding pair: two sashes on
    tracks either side of the glazing line, overlapping by an inch at the
    meeting stiles. One panel is a fixed pane in the outer frame."""
    fw, fd = frame_w, frame_d
    frame = [
        box(0.0, fw, -fd / 2, fd / 2, 0.0, height),
        box(length - fw, length, -fd / 2, fd / 2, 0.0, height),
        box(fw, length - fw, -fd / 2, fd / 2, height - fw, height),
        box(fw, length - fw, -fd / 2, fd / 2, 0.0, fw),
    ]
    glass = []
    if panels == 1:
        glass.append(box(fw, length - fw, -glass_t / 2, glass_t / 2, fw, height - fw))
    else:
        inch = 0.0254
        open_w = length - 2 * fw
        sash_w = open_w / 2 + inch / 2
        sd = min(fd / 2, 1.75 * inch)
        stile, top_rail, bottom_rail = 2.5 * inch, 2.5 * inch, 4.0 * inch
        z0, z1 = fw, height - fw
        for x0, y in ((fw, -fd / 4), (length - fw - sash_w, fd / 4)):
            x1 = x0 + sash_w
            y0, y1 = y - sd / 2, y + sd / 2
            frame += [
                box(x0, x0 + stile, y0, y1, z0, z1),
                box(x1 - stile, x1, y0, y1, z0, z1),
                box(x0 + stile, x1 - stile, y0, y1, z1 - top_rail, z1),
                box(x0 + stile, x1 - stile, y0, y1, z0, z0 + bottom_rail),
            ]
            t = min(glass_t, sd * 0.6)
            glass.append(box(x0 + stile, x1 - stile, y - t / 2, y + t / 2, z0 + bottom_rail, z1 - top_rail))
    return merge(frame), merge(glass)


def planar_faces(mesh: Mesh, tol: float = 1e-7) -> Mesh:
    """Split every face that is not flat into a fan of triangles."""
    faces, mats = [], []
    for f, m in zip(mesh.faces, mesh.face_mat or [0] * len(mesh.faces)):
        if len(f) > 3 and not is_planar(Mesh([mesh.verts[i] for i in f], [tuple(range(len(f)))]), tol):
            for i in range(1, len(f) - 1):
                faces.append((f[0], f[i], f[i + 1]))
                mats.append(m)
        else:
            faces.append(tuple(f))
            mats.append(m)
    return Mesh(list(mesh.verts), faces, mats)


def merge(meshes: Iterable[Mesh]) -> Mesh:
    out = Mesh()
    for m in meshes:
        out.add(m)
    return out


def ray_exit(poly: Sequence[Vec2], direction: Vec2) -> float:
    """Distance from the origin to where a ray leaves a convex polygon that
    contains the origin."""
    dx, dy = direction
    best = math.inf
    for i in range(len(poly)):
        (x0, y0), (x1, y1) = poly[i], poly[(i + 1) % len(poly)]
        ex, ey = x1 - x0, y1 - y0
        den = dx * ey - dy * ex
        if abs(den) < 1e-12:
            continue
        t = (x0 * ey - y0 * ex) / den
        s = (x0 * dy - y0 * dx) / den
        if t > 1e-12 and -1e-9 <= s <= 1 + 1e-9:
            best = min(best, t)
    if best is math.inf:
        raise ValueError("ray does not leave the polygon; is the origin inside?")
    return best
