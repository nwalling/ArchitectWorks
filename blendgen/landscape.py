"""The site round the building, for previews: graded ground, terrain, sky,
sun and planting. Runs inside Blender (bpy). Nothing here is exported to
the Unreal FBX set; Unreal builds its own landscape later.

The ground is a height field on a rectilinear grid: fine near the
building, coarse out to the far hills. Inside the building's outline it
drops away under the floors; on the pad round it it sits at finished grade;
beyond, it eases into invented rolling ground that falls the way the
drawings' grade lines do, rising to hills on the horizon.
"""

from __future__ import annotations

import math

import bpy
import numpy as np
from mathutils import Vector

SITE = "SITE"


def _smoothstep(a: float, b: float, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _seg_distance(px, py, rings) -> np.ndarray:
    """Distance from points to the outline, negative inside it."""
    d = np.full(px.shape, np.inf)
    inside = np.zeros(px.shape, bool)
    for ring in rings:
        r = np.asarray(ring, float)
        a, b = r[:-1], r[1:]
        for (ax, ay), (bx, by) in zip(a, b):
            ex, ey = bx - ax, by - ay
            L2 = ex * ex + ey * ey or 1e-12
            t = np.clip(((px - ax) * ex + (py - ay) * ey) / L2, 0, 1)
            d = np.minimum(d, np.hypot(px - (ax + t * ex), py - (ay + t * ey)))
            crosses = ((ay > py) != (by > py)) & (px < (bx - ax) * (py - ay) / ((by - ay) or 1e-12) + ax)
            inside ^= crosses
    return np.where(inside, -d, d)


class Terrain:
    def __init__(self, site: dict) -> None:
        t = site["terrain"]
        self.site, self.t = site, t
        pts = np.concatenate([np.asarray(r, float) for r in site["outline"]])
        self.cx, self.cy = pts[:, 0].mean(), pts[:, 1].mean()
        self.R0 = float(np.hypot(pts[:, 0] - self.cx, pts[:, 1] - self.cy).max())
        rng = np.random.default_rng(t["seed"])
        lam = t["wavelength_m"]
        self.waves = []
        for k in range(7):
            ang = rng.uniform(0, 2 * math.pi)
            wl = lam * rng.uniform(0.45, 1.6)
            self.waves.append((math.cos(ang) * 2 * math.pi / wl, math.sin(ang) * 2 * math.pi / wl, rng.uniform(0, 2 * math.pi), rng.uniform(0.5, 1.0)))
        self.amp = sum(w[3] for w in self.waves)
        self.hill_waves = []
        for k in range(5):
            ang = rng.uniform(0, 2 * math.pi)
            wl = rng.uniform(250.0, 700.0)
            self.hill_waves.append((math.cos(ang) * 2 * math.pi / wl, math.sin(ang) * 2 * math.pi / wl, rng.uniform(0, 2 * math.pi)))

    def height(self, X, Y, d=None):
        t, s = self.t, self.site
        if d is None:
            d = _seg_distance(X, Y, s["outline"])
        dx, dy = X - self.cx, Y - self.cy
        fall = t["fall"]
        base = -t["grade"] * (dx * fall[0] + dy * fall[1])
        noise = sum(a * np.sin(kx * X + ky * Y + ph) for kx, ky, ph, a in self.waves) / self.amp * t["relief_m"]
        r = np.hypot(dx, dy)
        E = t["extent_m"]
        hills = t["hills_m"] * _smoothstep(0.7 * E, 2.2 * E, r) * (0.6 + 0.4 * sum(np.sin(kx * X + ky * Y + ph) for kx, ky, ph in self.hill_waves) / len(self.hill_waves))
        w = _smoothstep(t["pad_m"], t["pad_m"] + t["blend_m"], d)
        z = s["grade_m"] + w * (base + noise + hills)
        # under the floors (and the sunken music room) the ground drops away;
        # the outline is 0.3 m outside the footprints, so the drop starts
        # 0.3 m inside them and the slope down stays hidden under the floors
        return np.where(d < -self.UNDER, s["grade_m"] - 0.75, z)

    UNDER = 0.6  # m inside the outline where the ground drops
    FINE = 0.2  # m grid spacing round the building

    def grid(self):
        """Grid lines: FINE round the building so the drop under it stays
        under it, 1 m out to the extent, then widening to the hills."""
        E = self.t["extent_m"]
        F = self.R0 + 6.0
        fine = np.arange(-F, F + 1e-6, self.FINE)
        mid = np.arange(np.ceil(F), E + 1e-6, 1.0)
        far = np.geomspace(E + 8.0, 6 * E, 28)
        axis = np.concatenate([-far[::-1], -mid[::-1], fine, mid, far])
        return self.cx + axis, self.cy + axis


def build_terrain(site: dict, coll, material) -> Terrain:
    T = Terrain(site)
    xs, ys = T.grid()
    X, Y = np.meshgrid(xs, ys)
    Z = T.height(X, Y)
    ny, nx = X.shape
    verts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    i = np.arange(ny - 1)[:, None] * nx + np.arange(nx - 1)[None, :]
    faces = np.stack([i, i + 1, i + nx + 1, i + nx], -1).reshape(-1, 4)
    me = bpy.data.meshes.new("SiteTerrain")
    me.from_pydata(verts.tolist(), [], faces.tolist())
    uv = me.uv_layers.new(name="UVMap")
    vidx = np.zeros(len(me.loops), np.int64)
    me.loops.foreach_get("vertex_index", vidx)
    uvs = verts[vidx][:, :2]  # metres: the shader scales to the texture's tile
    uv.data.foreach_set("uv", uvs.ravel())
    me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
    me.update()
    if material is not None:
        me.materials.append(material)
    ob = bpy.data.objects.new("SiteTerrain", me)
    coll.objects.link(ob)
    T.xs, T.ys, T.Z = xs, ys, Z
    return T


def sample_height(T: Terrain, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear height on the terrain grid."""
    i = np.clip(np.searchsorted(T.xs, x) - 1, 0, len(T.xs) - 2)
    j = np.clip(np.searchsorted(T.ys, y) - 1, 0, len(T.ys) - 2)
    fx = (x - T.xs[i]) / (T.xs[i + 1] - T.xs[i])
    fy = (y - T.ys[j]) / (T.ys[j + 1] - T.ys[j])
    z00, z10, z01, z11 = T.Z[j, i], T.Z[j, i + 1], T.Z[j + 1, i], T.Z[j + 1, i + 1]
    return (z00 * (1 - fx) + z10 * fx) * (1 - fy) + (z01 * (1 - fx) + z11 * fx) * fy


def in_view(px, py, views) -> np.ndarray:
    """Points between a camera and the building it looks at: inside the
    wedge from the camera (1.5 m wide there) to the building's span (plus
    3 m) at the target. ``views`` is [((cx, cy), (tx, ty), half_span)]."""
    hit = np.zeros(np.shape(px), bool)
    for (cx, cy), (tx, ty), half in views or []:
        ux, uy = tx - cx, ty - cy
        L = math.hypot(ux, uy) or 1e-9
        ux, uy = ux / L, uy / L
        along = (px - cx) * ux + (py - cy) * uy
        side = np.abs(-(px - cx) * uy + (py - cy) * ux)
        hit |= (along > -2.0) & (along < L) & (side < 1.5 + (half + 3.0) * np.clip(along, 0, None) / L)
    return hit


def scatter(T: Terrain, plant: dict, seed: int, views=None) -> np.ndarray:
    """Points (x, y) between ``near`` and ``far`` from the outline, at least
    ``spacing`` apart; returns as many as fit, up to ``count``. Spaced
    plants (trees, shrubs, rocks) keep out of the ``views`` (see in_view);
    ground cover, which has no spacing, does not, but thins out toward the
    far edge of its band."""
    rng = np.random.default_rng(seed)
    R = plant["far_m"] + T.R0
    want, sp = plant["count"], plant["spacing_m"]
    out = np.zeros((0, 2))
    for _ in range(40):
        n = max(64, (want - len(out)) * 6)
        p = np.stack([T.cx + rng.uniform(-R, R, n), T.cy + rng.uniform(-R, R, n)], 1)
        d = _seg_distance(p[:, 0], p[:, 1], T.site["outline"])
        band = (d >= plant["near_m"]) & (d <= plant["far_m"])
        p, d = p[band], d[band]
        if sp <= 0:
            # ground cover thins out over the outer part of its band rather
            # than stopping along a line
            near, far = plant["near_m"], plant["far_m"]
            p = p[rng.uniform(0, 1, len(p)) < 1.0 - _smoothstep(near + 0.5 * (far - near), far, d)]
        if sp > 0 and views:
            p = p[~in_view(p[:, 0], p[:, 1], views)]
        if sp > 0:
            keep = []
            for q in p:
                if len(out) + len(keep) >= want:
                    break
                others = np.vstack([out, np.asarray(keep).reshape(-1, 2)]) if len(keep) else out
                if len(others) == 0 or np.min(np.hypot(*(others - q).T)) >= sp:
                    keep.append(q)
            out = np.vstack([out, np.asarray(keep).reshape(-1, 2)])
        else:
            out = np.vstack([out, p[: want - len(out)]])
        if len(out) >= want:
            break
    return out[:want]


def _instancer_tree(name: str, source) -> bpy.types.NodeTree:
    ng = bpy.data.node_groups.new(name, "GeometryNodeTree")
    ng.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    n, link = ng.nodes, ng.links.new
    gi, go = n.new("NodeGroupInput"), n.new("NodeGroupOutput")
    inst = n.new("GeometryNodeInstanceOnPoints")
    coll = n.new("GeometryNodeCollectionInfo")
    coll.inputs["Collection"].default_value = source
    coll.inputs["Separate Children"].default_value = True
    coll.inputs["Reset Children"].default_value = True
    coll.transform_space = "ORIGINAL"
    rot, scl, var = (n.new("GeometryNodeInputNamedAttribute") for _ in range(3))
    rot.data_type = scl.data_type = "FLOAT"
    var.data_type = "INT"
    rot.inputs["Name"].default_value, scl.inputs["Name"].default_value, var.inputs["Name"].default_value = "aw_rot", "aw_scale", "aw_variant"
    euler = n.new("ShaderNodeCombineXYZ")
    link(rot.outputs["Attribute"], euler.inputs["Z"])
    link(gi.outputs["Geometry"], inst.inputs["Points"])
    link(coll.outputs["Instances"], inst.inputs["Instance"])
    inst.inputs["Pick Instance"].default_value = True
    link(var.outputs["Attribute"], inst.inputs["Instance Index"])
    link(euler.outputs["Vector"], inst.inputs["Rotation"])
    link(scl.outputs["Attribute"], inst.inputs["Scale"])
    link(inst.outputs["Instances"], go.inputs["Geometry"])
    return ng


def set_controls(source, controls: dict) -> list[str]:
    """Set the model's own material controls: unlinked inputs of the node
    groups in its materials, by name. Returns the inputs set."""
    done, hit = set(), []
    for ob in source.all_objects:
        for slot in getattr(ob, "material_slots", []):
            mat = slot.material
            if mat is None or mat.name in done or mat.node_tree is None:
                continue
            done.add(mat.name)
            for node in mat.node_tree.nodes:
                if node.type != "GROUP":
                    continue
                for name, value in controls.items():
                    sock = node.inputs.get(name)
                    if sock is not None and not sock.is_linked:
                        sock.default_value = value
                        hit.append(f"{mat.name}.{name}")
    return hit


def plant(T: Terrain, coll, sources_parent, model_blend: str, spec: dict, seed: int, log, views=None) -> int:
    mid = spec["polyhaven"]
    with bpy.data.libraries.load(model_blend, link=False) as (src, dst):
        want = next((c for c in (f"{mid}_LOD0", f"{mid}_static", mid) if c in src.collections), None)
        if want is None:
            log(f"site: {mid} has none of the expected collections; skipped")
            return 0
        dst.collections = [want]
    source = dst.collections[0]
    source.name = f"SITE_SRC_{mid}"
    if spec.get("controls"):
        hit = set_controls(source, spec["controls"])
        if not hit:
            log(f"site: {mid} has no material inputs named {sorted(spec['controls'])}")
    sources_parent.children.link(source)
    variants = len(source.objects)
    pts = scatter(T, spec, seed, views)
    if len(pts) == 0:
        return 0
    rng = np.random.default_rng(seed + 1)
    z = sample_height(T, pts[:, 0], pts[:, 1])
    me = bpy.data.meshes.new(f"Plant_{mid}")
    me.from_pydata(np.column_stack([pts, z]).tolist(), [], [])
    lo, hi = spec["scale"]
    for nm, typ, vals in (
        ("aw_rot", "FLOAT", rng.uniform(0, 2 * math.pi, len(pts))),
        ("aw_scale", "FLOAT", rng.uniform(lo, hi, len(pts))),
        ("aw_variant", "INT", rng.integers(0, max(1, variants), len(pts))),
    ):
        a = me.attributes.new(nm, typ, "POINT")
        a.data.foreach_set("value", vals.tolist())
    ob = bpy.data.objects.new(f"Plant_{mid}", me)
    coll.objects.link(ob)
    mod = ob.modifiers.new("Instances", "NODES")
    mod.node_group = _instancer_tree(f"AW_Instance_{mid}", source)
    return len(pts)


def make_paths_absolute() -> None:
    for img in bpy.data.images:
        if img.filepath and not img.packed_file:
            img.filepath = bpy.path.abspath(img.filepath, library=img.library)


def set_sky(site: dict, strength: float = 0.12):
    scene = bpy.context.scene
    world = bpy.data.worlds.new("SiteSky")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    bg = nt.nodes["Background"]
    sky = nt.nodes.new("ShaderNodeTexSky")
    for kind in ("MULTIPLE_SCATTERING", "SINGLE_SCATTERING", "HOSEK_WILKIE"):
        try:
            sky.sky_type = kind
            break
        except TypeError:
            continue
    sun = site["sun"]
    d = Vector(sun["direction"])
    if hasattr(sky, "sun_disc"):
        sky.sun_disc = False
    if hasattr(sky, "sun_elevation"):
        sky.sun_elevation = math.radians(sun["elevation_deg"])
        # Blender measures the sky's sun rotation from +Y toward -X
        sky.sun_rotation = math.atan2(-d.x, d.y) % (2 * math.pi)
    if hasattr(sky, "sun_direction"):
        try:
            sky.sun_direction = d
        except (AttributeError, TypeError):
            pass
    nt.links.new(sky.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = strength
    lamp = bpy.data.objects.new("SiteSun", bpy.data.lights.new("SiteSun", "SUN"))
    lamp.data.energy = 3.4
    lamp.data.angle = math.radians(0.6)
    lamp.data.color = (1.0, 0.95, 0.86)
    lamp.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()
    scene.collection.objects.link(lamp)
    return lamp


def build_site(site: dict, assets: dict | None, materials: dict, log, views=None) -> dict:
    scene = bpy.context.scene
    coll = bpy.data.collections.new(SITE)
    scene.collection.children.link(coll)
    sources = bpy.data.collections.new("SITE_SOURCES")
    scene.collection.children.link(sources)
    ground = materials.get(site["terrain"]["ground"])
    T = build_terrain(site, coll, ground)
    counts = {}
    models = (assets or {}).get("models", {})
    for k, p in enumerate(site["planting"]):
        blend = models.get(p["polyhaven"])
        if not blend:
            counts[p["polyhaven"]] = "not fetched"
            continue
        counts[p["polyhaven"]] = plant(T, coll, sources, blend, p, site["terrain"]["seed"] * 1000 + k, log, views)
    make_paths_absolute()
    for lc in bpy.context.view_layer.layer_collection.children:
        if lc.name == "SITE_SOURCES":
            lc.exclude = True
    set_sky(site)
    scene = bpy.context.scene
    for vt in ("AgX", "Filmic", "Standard"):
        try:
            scene.view_settings.view_transform = vt
            break
        except TypeError:
            continue
    scene.view_settings.exposure = 0.3
    log(f"site: terrain {len(T.xs)}x{len(T.ys)} grid; planting {counts}")
    return {"terrain_grid": [len(T.xs), len(T.ys)], "planting": counts, "sun": site["sun"]}
