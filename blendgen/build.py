"""Build the .blend from a resolved model, export the Unreal meshes, verify.

Runs inside Blender and needs nothing beyond Blender's own Python, so either
of these works:

    blender --background --python blendgen/build.py -- out/resolved.json out/
    python blendgen/build.py out/resolved.json out/        # with the bpy wheel

The script makes no geometry decisions: every vertex arrives in resolved.json.
It lays out collections, shares mesh data between placements, attaches an
FBX exporter to each component collection, exports, then re-imports every FBX
into an empty file and checks it against the resolved model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

import bpy
from mathutils import Matrix, Vector

FBX_SETTINGS = {
    # Unit scale stays 1.0 in Blender (metres). The FBX carries a unit scale
    # factor instead, so objects keep scale 1 and Unreal converts to cm.
    "apply_scale_options": "FBX_SCALE_UNITS",
    "apply_unit_scale": True,
    "global_scale": 1.0,
    "axis_forward": "-Z",
    "axis_up": "Y",
    "object_types": {"MESH"},
    "use_mesh_modifiers": True,
    "mesh_smooth_type": "FACE",
    "use_triangles": True,
    "use_tspace": False,
    "use_custom_props": False,
    "add_leaf_bones": False,
    "bake_anim": False,
    "embed_textures": False,
    "path_mode": "AUTO",
}
COMPONENTS = "AW_COMPONENTS"
BUILDING = "BUILDING"
ASSUMED = "ASSUMED"
REFERENCE = "REFERENCE"


def log(msg: str) -> None:
    print(f"[blendgen] {msg}", flush=True)


# ---- scene ---------------------------------------------------------------


def reset_scene() -> bpy.types.Scene:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    scene.unit_settings.length_unit = "METERS"
    return scene


def srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def rubble(mat: bpy.types.Material, rgb: tuple[float, float, float]) -> None:
    """Random rubble masonry for previews: stones about 0.35 m across from a
    Voronoi pattern in world space, each a little different in tone, with
    sunken mortar joints. Unreal gets only the material slot."""
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links
    bsdf = nodes.get("Principled BSDF")
    geo = nodes.new("ShaderNodeNewGeometry")
    mapping = nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (2.6, 2.6, 3.4)
    links.new(geo.outputs["Position"], mapping.inputs["Vector"])
    edge = nodes.new("ShaderNodeTexVoronoi")
    edge.feature = "DISTANCE_TO_EDGE"
    edge.inputs["Scale"].default_value = 1.0
    links.new(mapping.outputs["Vector"], edge.inputs["Vector"])
    cells = nodes.new("ShaderNodeTexVoronoi")
    cells.inputs["Scale"].default_value = 1.0
    links.new(mapping.outputs["Vector"], cells.inputs["Vector"])
    joint = nodes.new("ShaderNodeMapRange")
    joint.inputs["From Min"].default_value = 0.02
    joint.inputs["From Max"].default_value = 0.07
    links.new(edge.outputs["Distance"], joint.inputs["Value"])
    # each stone a little lighter or darker than the spec colour, never another hue
    grey = nodes.new("ShaderNodeRGBToBW")
    links.new(cells.outputs["Color"], grey.inputs["Color"])
    shade = nodes.new("ShaderNodeMapRange")
    shade.inputs["To Min"].default_value = 0.72
    shade.inputs["To Max"].default_value = 1.22
    links.new(grey.outputs["Val"], shade.inputs["Value"])
    tone = nodes.new("ShaderNodeMix")
    tone.data_type = "RGBA"
    tone.blend_type = "MULTIPLY"
    tone.inputs["Factor"].default_value = 1.0
    tone.inputs["A"].default_value = (*rgb, 1.0)
    links.new(shade.outputs["Result"], tone.inputs["B"])
    stone = nodes.new("ShaderNodeMix")
    stone.data_type = "RGBA"
    stone.inputs["A"].default_value = (0.32, 0.30, 0.27, 1.0)  # mortar
    links.new(joint.outputs["Result"], stone.inputs["Factor"])
    links.new(tone.outputs["Result"], stone.inputs["B"])
    links.new(stone.outputs["Result"], bsdf.inputs["Base Color"])
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.6
    bump.inputs["Distance"].default_value = 0.02
    links.new(joint.outputs["Result"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    bsdf.inputs["Roughness"].default_value = 0.9


def shingles(mat: bpy.types.Material, rgb: tuple[float, float, float]) -> None:
    """Composition tab shingles for previews: level courses about 5" to the
    weather (2 1/2" of height on these slopes) and 12" tabs, laid round the
    object's own origin, which for a conical roof is the pavilion's centre."""
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links
    bsdf = nodes.get("Principled BSDF")
    tc = nodes.new("ShaderNodeTexCoord")
    xyz = nodes.new("ShaderNodeSeparateXYZ")
    links.new(tc.outputs["Object"], xyz.inputs["Vector"])
    ang = nodes.new("ShaderNodeMath")
    ang.operation = "ARCTAN2"
    links.new(xyz.outputs["Y"], ang.inputs[0])
    links.new(xyz.outputs["X"], ang.inputs[1])
    flat = nodes.new("ShaderNodeCombineXYZ")
    links.new(xyz.outputs["X"], flat.inputs["X"])
    links.new(xyz.outputs["Y"], flat.inputs["Y"])
    rad = nodes.new("ShaderNodeVectorMath")
    rad.operation = "LENGTH"
    links.new(flat.outputs["Vector"], rad.inputs[0])
    arc = nodes.new("ShaderNodeMath")
    arc.operation = "MULTIPLY"
    links.new(ang.outputs["Value"], arc.inputs[0])
    links.new(rad.outputs["Value"], arc.inputs[1])
    s_ = nodes.new("ShaderNodeMath")
    s_.operation = "MULTIPLY"
    s_.inputs[1].default_value = 0.5 / 0.305  # 12" tabs
    links.new(arc.outputs["Value"], s_.inputs[0])
    t_ = nodes.new("ShaderNodeMath")
    t_.operation = "MULTIPLY"
    t_.inputs[1].default_value = 0.25 / 0.0635  # a course every 2 1/2" of height
    links.new(xyz.outputs["Z"], t_.inputs[0])
    uv = nodes.new("ShaderNodeCombineXYZ")
    links.new(s_.outputs["Value"], uv.inputs["X"])
    links.new(t_.outputs["Value"], uv.inputs["Y"])
    brick = nodes.new("ShaderNodeTexBrick")
    brick.offset = 0.5
    brick.offset_frequency = 2
    brick.inputs["Scale"].default_value = 1.0
    brick.inputs["Mortar Size"].default_value = 0.006
    brick.inputs["Brick Width"].default_value = 0.5
    brick.inputs["Row Height"].default_value = 0.25
    brick.inputs["Color1"].default_value = (*rgb, 1.0)
    brick.inputs["Color2"].default_value = (rgb[0] * 0.78, rgb[1] * 0.78, rgb[2] * 0.78, 1.0)
    brick.inputs["Mortar"].default_value = (rgb[0] * 0.35, rgb[1] * 0.35, rgb[2] * 0.35, 1.0)
    links.new(uv.outputs["Vector"], brick.inputs["Vector"])
    links.new(brick.outputs["Color"], bsdf.inputs["Base Color"])
    inv = nodes.new("ShaderNodeMath")
    inv.operation = "SUBTRACT"
    inv.inputs[0].default_value = 1.0
    links.new(brick.outputs["Fac"], inv.inputs[1])
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.5
    bump.inputs["Distance"].default_value = 0.01
    links.new(inv.outputs["Value"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    bsdf.inputs["Roughness"].default_value = 0.85


def make_materials(materials: dict) -> dict[str, bpy.types.Material]:
    """Flat-colour placeholders, or a procedural pattern where the spec names
    one. Spec colours are sRGB; Blender wants linear."""
    out = {}
    for name, m in materials.items():
        mat = bpy.data.materials.new(name)
        r, g, b = (srgb_to_linear(c) for c in m["color"][:3])
        a = m["color"][3]
        mat.diffuse_color = (r, g, b, a)
        try:
            mat.use_nodes = True
        except AttributeError:
            pass
        bsdf = mat.node_tree.nodes.get("Principled BSDF") if mat.node_tree else None
        if bsdf is not None:
            bsdf.inputs["Base Color"].default_value = (r, g, b, 1.0)
            bsdf.inputs["Roughness"].default_value = 0.15 if m["translucent"] else 0.7
            if m["translucent"]:
                bsdf.inputs["Alpha"].default_value = a
            pattern = {"rubble": rubble, "shingles": shingles}.get(m.get("pattern"))
            if pattern is not None:
                try:
                    pattern(mat, (r, g, b))
                except (KeyError, AttributeError, RuntimeError) as err:  # node names differ between Blender versions
                    log(f"warning: no {m['pattern']} pattern for {name}: {err}")
        if m["translucent"]:
            for attr, value in (("surface_render_method", "BLENDED"), ("blend_method", "BLEND")):
                try:
                    setattr(mat, attr, value)
                    break
                except (AttributeError, TypeError):
                    continue
        out[name] = mat
    return out


def new_collection(name: str, parent: bpy.types.Collection) -> bpy.types.Collection:
    c = bpy.data.collections.new(name)
    parent.children.link(c)
    return c


def build_mesh(name: str, data: dict, materials: list[bpy.types.Material] | None) -> bpy.types.Mesh:
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in data["verts"]], [], [tuple(f) for f in data["faces"]])
    if materials:
        for mat in materials:
            me.materials.append(mat)
        me.polygons.foreach_set("material_index", data["face_mat"])
    if "uv" in data:
        layer = me.uv_layers.new(name="UVMap")
        flat = [c for face in data["uv"] for uv in face for c in uv]
        layer.data.foreach_set("uv", flat)
    if me.validate(clean_customdata=False):
        log(f"warning: mesh {name} needed cleanup")
    me.update()
    return me


def add_exporter(coll: bpy.types.Collection, rel_path: str) -> None:
    ex = coll.exporters.new("IO_FH_fbx", name="Unreal FBX")
    ex.filepath = rel_path
    props = ex.export_properties
    for key, value in FBX_SETTINGS.items():
        try:
            setattr(props, key, value)
        except (AttributeError, TypeError) as err:
            log(f"warning: FBX setting {key} not applied: {err}")


def build(resolved: dict, out_dir: str) -> dict:
    scene = reset_scene()
    mats = make_materials(resolved["materials"])

    comp_root = new_collection(COMPONENTS, scene.collection)
    building = new_collection(BUILDING, scene.collection)
    cat_colls = {c: new_collection(c, building) for c in resolved["categories"]}
    assumed = new_collection(ASSUMED, scene.collection)

    meshes = {}
    for comp in resolved["components"]:
        name = comp["name"]
        coll = new_collection(name, comp_root)
        me = build_mesh(name, comp["mesh"], [mats[m] for m in comp["material_slots"]])
        meshes[name] = me
        master = bpy.data.objects.new(name, me)
        coll.objects.link(master)
        for i, hull in enumerate(comp["collision"]):
            hm = build_mesh(f"UCX_{name}_{i:02d}", hull, None)
            ho = bpy.data.objects.new(f"UCX_{name}_{i:02d}", hm)
            ho.display_type = "WIRE"
            ho.hide_render = True
            coll.objects.link(ho)
        add_exporter(coll, f"//unreal/meshes/{name}.fbx")
        # Hidden from the viewport and renders, but not excluded from the view
        # layer: File > Export All Collections skips excluded collections.
        coll.hide_viewport = True
        coll.hide_render = True
        coll["aw_component"] = name
    comp_root.hide_render = True

    for p in resolved["placements"]:
        ob = bpy.data.objects.new(p["id"], meshes[p["component"]])
        ob.matrix_world = Matrix(p["matrix"])
        cat_colls[p["category"]].objects.link(ob)
        if p["basis"] == "assumed":
            assumed.objects.link(ob)
        ob["aw_element"] = p["element"]
        ob["aw_component"] = p["component"]
        ob["aw_basis"] = p["basis"]
        ob["aw_confidence"] = p["confidence"]
        ob["aw_source"] = json.dumps(p["source"], sort_keys=True)

    refs = resolved.get("references", [])
    if refs:
        ref_coll = new_collection(REFERENCE, scene.collection)
        ref_coll.hide_render = True
        for ref in refs:
            add_reference_plane(ref_coll, ref)

    scene["aw_building"] = resolved["building"]["id"]
    scene["aw_generator"] = resolved["generator"]
    path = os.path.join(out_dir, "generated.blend")
    bpy.ops.wm.save_as_mainfile(filepath=path, relative_remap=True)
    log(f"saved {path}")
    return {"blend": path, "objects": len(resolved["placements"]), "components": len(resolved["components"])}


def add_reference_plane(coll: bpy.types.Collection, ref: dict) -> None:
    """A straightened sheet lying under the model at true scale."""
    (x0, y0), (x1, y1), z = ref["min"], ref["max"], ref["z"]
    me = bpy.data.meshes.new(f"REF_{ref['sheet']}")
    me.from_pydata([(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)], [], [(0, 1, 2, 3)])
    me.uv_layers.new(name="UVMap").data.foreach_set("uv", [0, 0, 1, 0, 1, 1, 0, 1])
    mat = bpy.data.materials.new(f"REF_{ref['sheet']}")
    try:
        mat.use_nodes = True
    except AttributeError:
        pass
    nodes = mat.node_tree.nodes
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(ref["image"], check_existing=True)
    bsdf = nodes.get("Principled BSDF")
    if bsdf is not None:
        mat.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    me.materials.append(mat)
    ob = bpy.data.objects.new(f"REF_{ref['sheet']}", me)
    ob.hide_select = True
    ob["aw_reference"] = ref["sheet"]
    coll.objects.link(ob)


# ---- export ----------------------------------------------------------------


def sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def export_meshes(resolved: dict, out_dir: str) -> dict:
    mesh_dir = os.path.join(out_dir, "unreal", "meshes")
    os.makedirs(mesh_dir, exist_ok=True)  # the exporter does not create folders
    expected = {c["name"]: os.path.join(mesh_dir, c["name"] + ".fbx") for c in resolved["components"]}
    for f in os.listdir(mesh_dir):
        if f.endswith(".fbx"):
            os.remove(os.path.join(mesh_dir, f))
    started = time.time()
    mode = "collection_export_all"
    bpy.ops.wm.collection_export_all()
    # The operator reports FINISHED even when a write fails, so check the files.
    missing = [n for n, p in expected.items() if not os.path.isfile(p) or os.path.getmtime(p) < started - 1]
    if missing:
        log(f"Export All left {len(missing)} files unwritten; exporting those collections one by one")
        mode = "exporter_export_fallback"
        for name in missing:
            coll = bpy.data.collections[name]
            with bpy.context.temp_override(collection=coll):
                bpy.ops.collection.exporter_export(index=0)
        missing = [n for n, p in expected.items() if not os.path.isfile(p)]
    files = [
        {"component": n, "file": os.path.relpath(p, os.path.join(out_dir, "unreal")), "bytes": os.path.getsize(p), "sha1": sha1(p)}
        for n, p in expected.items()
        if os.path.isfile(p)
    ]
    return {"mode": mode, "files": files, "missing": missing, "settings": {k: sorted(v) if isinstance(v, set) else v for k, v in FBX_SETTINGS.items()}}


def export_preview_glb(resolved: dict, out_dir: str) -> str:
    path = os.path.join(out_dir, "preview", resolved["building"]["id"] + ".glb")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    for coll in bpy.data.collections[BUILDING].children:
        for ob in coll.objects:
            ob.select_set(True)
    bpy.ops.export_scene.gltf(filepath=path, export_format="GLB", use_selection=True)
    return path


def render_preview(resolved: dict, out_dir: str, samples: int) -> str:
    scene = bpy.context.scene
    pts = []
    for p in resolved["placements"]:
        if p["category"] == "Fixtures":
            continue
        m = p["matrix"]
        pts.append((m[0][3], m[1][3], m[2][3]))
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    span = max(max(xs) - min(xs), max(ys) - min(ys), 7.7)
    target = Vector((cx, cy, 1.6))

    cam_data = bpy.data.cameras.new("PreviewCamera")
    cam_data.lens = 30
    cam = bpy.data.objects.new("PreviewCamera", cam_data)
    scene.collection.objects.link(cam)
    cam.location = target + Vector((9.0, 11.5, 4.2)) * max(1.0, 0.72 * span / 7.7)
    cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()
    scene.camera = cam

    sun_data = bpy.data.lights.new("PreviewSun", "SUN")
    sun_data.energy = 2.2
    sun_data.angle = math.radians(3)
    sun = bpy.data.objects.new("PreviewSun", sun_data)
    sun.rotation_euler = (math.radians(48), math.radians(0), math.radians(140))
    scene.collection.objects.link(sun)

    ground = bpy.data.meshes.new("PreviewGround")
    g0 = 6 * span
    ground.from_pydata([(cx - g0, cy - g0, -0.11), (cx + g0, cy - g0, -0.11), (cx + g0, cy + g0, -0.11), (cx - g0, cy + g0, -0.11)], [], [(0, 1, 2, 3)])
    gmat = bpy.data.materials.new("PreviewGround")
    gmat.diffuse_color = (0.42, 0.40, 0.34, 1.0)
    try:
        gmat.use_nodes = True
    except AttributeError:
        pass
    bsdf = gmat.node_tree.nodes.get("Principled BSDF") if gmat.node_tree else None
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = (0.42, 0.40, 0.34, 1.0)
    ground.materials.append(gmat)
    scene.collection.objects.link(bpy.data.objects.new("PreviewGround", ground))

    world = bpy.data.worlds.new("PreviewWorld")
    scene.world = world
    try:
        world.use_nodes = True
    except AttributeError:
        pass
    bg = world.node_tree.nodes.get("Background") if world.node_tree else None
    if bg is not None:
        bg.inputs["Color"].default_value = (0.78, 0.83, 0.88, 1.0)
        bg.inputs["Strength"].default_value = 0.45

    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = samples
    try:
        scene.cycles.use_denoising = True
    except AttributeError:
        pass
    scene.render.resolution_x, scene.render.resolution_y = 1600, 1000
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.exposure = -0.2
    stem = os.path.join(out_dir, "preview", resolved["building"]["id"])
    scene.render.filepath = stem + ".png"
    bpy.ops.render.render(write_still=True)

    # the other side, where the stone walls are
    cam.location = target + Vector((-9.5, -11.0, 4.6)) * max(1.0, 0.72 * span / 7.7)
    cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()
    scene.render.filepath = stem + "-stone-side.png"
    bpy.ops.render.render(write_still=True)

    # straight down, orthographic, north (plan +Y) up: compare with the plan sheet
    top_data = bpy.data.cameras.new("PreviewTop")
    top_data.type = "ORTHO"
    top_data.ortho_scale = span * 1.35
    top = bpy.data.objects.new("PreviewTop", top_data)
    scene.collection.objects.link(top)
    top.location = (cx, cy, 40.0)
    top.rotation_euler = (0.0, 0.0, 0.0)
    scene.camera = top
    scene.render.resolution_x, scene.render.resolution_y = 1200, 1200
    scene.render.filepath = stem + "-top.png"
    bpy.ops.render.render(write_still=True)
    return stem + ".png"


AXES = {"+X": (1.0, 0.0, 0.0), "-X": (-1.0, 0.0, 0.0), "+Y": (0.0, 1.0, 0.0), "-Y": (0.0, -1.0, 0.0)}


def building_corners() -> list[Vector]:
    out = []
    for coll in bpy.data.collections[BUILDING].children:
        for ob in coll.objects:
            if ob.type == "MESH" and "handedness" not in ob.name.lower():
                out.extend(ob.matrix_world @ Vector(c) for c in ob.bound_box)
    return out


def render_elevations(resolved: dict, out_dir: str, looking: list[str], samples: int, width_px: int = 1800) -> list[dict]:
    """Orthographic elevations, one per viewing direction, on a transparent
    background. Each comes with the mapping from its pixels to plan and
    height (metres), so a drawn elevation can be laid over it."""
    scene = bpy.context.scene
    ground = bpy.data.objects.get("PreviewGround")
    if ground is not None:
        ground.hide_render = True
    test_pieces = [ob for ob in bpy.data.objects if "handedness" in ob.name.lower()]
    for ob in test_pieces:
        ob.hide_render = True
    scene.render.film_transparent = True
    scene.cycles.samples = max(16, samples // 2)
    corners = building_corners()
    out = []
    for key in looking:
        L = Vector(AXES[key])
        H = L.cross(Vector((0.0, 0.0, 1.0)))
        hs = [c.dot(H) for c in corners]
        zs = [c.z for c in corners]
        ds = [c.dot(L) for c in corners]
        pad = 1.0
        h0, h1 = min(hs) - pad, max(hs) + pad
        z0, z1 = min(zs) - pad, max(zs) + pad
        ppm = width_px / (h1 - h0)
        res_y = int(round((z1 - z0) * ppm))
        z0 = z1 - res_y / ppm
        data = bpy.data.cameras.new(f"Elevation{key}")
        data.type = "ORTHO"
        data.sensor_fit = "HORIZONTAL"
        data.ortho_scale = h1 - h0
        data.clip_end = 1000.0
        cam = bpy.data.objects.new(f"Elevation{key}", data)
        scene.collection.objects.link(cam)
        centre = H * ((h0 + h1) / 2) + Vector((0.0, 0.0, (z0 + z1) / 2)) + L * (min(ds) - 50.0)
        cam.location = centre
        cam.rotation_euler = L.to_track_quat("-Z", "Y").to_euler()
        scene.camera = cam
        scene.render.resolution_x, scene.render.resolution_y = width_px, res_y
        name = f"elevation-looking{key.replace('+', 'p').replace('-', 'm')}.png"
        scene.render.filepath = os.path.join(out_dir, "preview", name)
        bpy.ops.render.render(write_still=True)
        out.append({"looking": key, "file": os.path.join("preview", name), "h_axis": [H.x, H.y, H.z], "h0_m": h0, "z_top_m": z1, "px_per_m": ppm, "size": [width_px, res_y]})
    scene.render.film_transparent = False
    if ground is not None:
        ground.hide_render = False
    for ob in test_pieces:
        ob.hide_render = False
    return out


# ---- verification -------------------------------------------------------------


def fbx_global_settings(path: str) -> dict:
    """Read the FBX header the way an importer sees it."""
    try:
        from io_scene_fbx import parse_fbx
    except ImportError:
        return {}
    root, _version = parse_fbx.parse(path)
    out = {}
    for elem in root.elems:
        if elem.id != b"GlobalSettings":
            continue
        for sub in elem.elems:
            if sub.id != b"Properties70":
                continue
            for p in sub.elems:
                key = p.props[0].decode() if isinstance(p.props[0], bytes) else str(p.props[0])
                if key in {"UpAxis", "UpAxisSign", "FrontAxis", "FrontAxisSign", "CoordAxis", "CoordAxisSign", "UnitScaleFactor", "OriginalUnitScaleFactor"}:
                    out[key] = p.props[4]
    return out


def verify(resolved: dict, out_dir: str, tol: float = 1e-4) -> dict:
    results = []
    for comp in resolved["components"]:
        name = comp["name"]
        path = os.path.join(out_dir, "unreal", "meshes", name + ".fbx")
        res = {"component": name, "ok": False}
        results.append(res)
        if not os.path.isfile(path):
            res["error"] = "file missing"
            continue
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.wm.fbx_import(filepath=path)
        objs = {o.name: o for o in bpy.data.objects}
        main = objs.get(name)
        ucx = sorted(n for n in objs if n.startswith(f"UCX_{name}_"))
        if main is None:
            res["error"] = f"no object named {name}; found {sorted(objs)}"
            continue
        world = [main.matrix_world @ v.co for v in main.data.vertices]
        bmin = [min(v[i] for v in world) for i in range(3)]
        bmax = [max(v[i] for v in world) for i in range(3)]
        dmin = max(abs(a - b) for a, b in zip(bmin, comp["bounds"]["min"]))
        dmax = max(abs(a - b) for a, b in zip(bmax, comp["bounds"]["max"]))
        main.data.calc_loop_triangles()
        tris = len(main.data.loop_triangles)
        slots = [m.name for m in main.data.materials if m]
        res.update(
            {
                "bounds_error_m": round(max(dmin, dmax), 7),
                "scale": [round(s, 6) for s in main.scale],
                "triangles": tris,
                "expected_triangles": comp["triangles"],
                "ucx": len(ucx),
                "expected_ucx": len(comp["collision"]),
                "material_slots": slots,
            }
        )
        problems = []
        if max(dmin, dmax) > tol:
            problems.append("bounds differ after re-import")
        if tris != comp["triangles"]:
            problems.append("triangle count differs")
        if len(ucx) != len(comp["collision"]):
            problems.append("collision hull count differs")
        if any(abs(s - 1.0) > 1e-6 for s in main.scale):
            problems.append("object scale is not 1")
        if not all(any(s.startswith(m) for s in slots) for m in comp["material_slots"]):
            problems.append("material slot missing")
        res["problems"] = problems
        res["ok"] = not problems
    sample = os.path.join(out_dir, "unreal", "meshes", resolved["components"][0]["name"] + ".fbx")
    return {"all_ok": all(r["ok"] for r in results), "components": results, "fbx_global_settings": fbx_global_settings(sample) if os.path.isfile(sample) else {}}


# ---- entry point ---------------------------------------------------------------


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="blendgen")
    ap.add_argument("resolved")
    ap.add_argument("out_dir")
    ap.add_argument("--preview", action="store_true", help="render a Cycles preview image")
    ap.add_argument("--samples", type=int, default=48)
    ap.add_argument("--elevations", default="", help="also render orthographic elevations looking along these axes, e.g. -X,+Y")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args(argv)

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    with open(args.resolved, encoding="utf-8") as f:
        resolved = json.load(f)

    report = {"blender": bpy.app.version_string, "generator": resolved["generator"], "background": bpy.app.background}
    report["build"] = build(resolved, out_dir)
    report["export"] = export_meshes(resolved, out_dir)
    report["preview_glb"] = os.path.relpath(export_preview_glb(resolved, out_dir), out_dir)
    if args.preview:
        report["preview_png"] = os.path.relpath(render_preview(resolved, out_dir, args.samples), out_dir)
        looking = [k for k in args.elevations.split(",") if k]
        if looking:
            report["elevations"] = render_elevations(resolved, out_dir, looking, args.samples)
    if not args.no_verify:
        report["verify"] = verify(resolved, out_dir)
    ok = not report["export"]["missing"] and report.get("verify", {}).get("all_ok", True)
    report["ok"] = ok
    with open(os.path.join(out_dir, "blender_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    log("report written; " + ("all checks passed" if ok else "PROBLEMS FOUND"))
    return 0 if ok else 1


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    sys.exit(main(argv))
