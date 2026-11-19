"""The phase 0 import test: does the model in Unreal match the spec?

Run after import_unreal.py, the same way (py "<bundle>/unreal/check_import.py").
It writes check_report.json next to the manifest and logs PASS or FAIL.

What it asserts, all against numbers computed outside Unreal:

1. Every mesh asset exists and its local bounds match (size, units, axes).
   The handedness marker is an L with a post, different along every axis,
   so a mirrored, swapped or rotated import is caught and named.
2. Every mesh that should have collision has the expected number of convex
   hulls, and Nanite is on exactly where the manifest says.
3. Every placement has an actor, and each actor's world bounds match.
   Rotated actors make this sensitive to rotation errors, not just position.
4. Instance counts per mesh match, and the whole building's bounds match.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter

import unreal


def script_dir() -> str:
    f = globals().get("__file__") or (sys.argv[0] if sys.argv and sys.argv[0].endswith(".py") else None)
    return os.path.dirname(os.path.abspath(f)) if f else os.getcwd()


def bundle_dir() -> str:
    args = [a for a in sys.argv[1:] if not a.endswith(".py") and not a.startswith("--")]
    if args:
        return os.path.abspath(args[0])
    return os.path.abspath(os.environ.get("AW_BUNDLE", script_dir()))


def vec(v) -> list[float]:
    return [round(v.x, 3), round(v.y, 3), round(v.z, 3)]


def close(a, b, tol: float) -> bool:
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def diagnose_marker(got_min, got_max, exp_min, exp_max) -> str:
    ext = [b - a for a, b in zip(got_min, got_max)]
    exp = [b - a for a, b in zip(exp_min, exp_max)]
    if close(got_min, exp_min, 0.5) and close(got_max, exp_max, 0.5):
        return "marker matches: Blender +X -> Unreal +X, Blender +Y -> Unreal -Y, Z up, metres -> cm"
    if close(ext, exp, 0.5) and close(got_min, [exp_min[0], 0.0, exp_min[2]], 0.5):
        return "marker is not mirrored: the importer kept Blender's +Y, so the manifest's Y flip is wrong for this setup"
    if close(ext, [exp[1], exp[0], exp[2]], 0.5):
        return "X and Y are swapped: 'Force Front X Axis' is probably on in the FBX import settings"
    ratio = [g / e if e else 0 for g, e in zip(ext, exp)]
    if all(abs(r - 0.01) < 0.002 for r in ratio):
        return "marker is 100x too small: scene unit conversion is off"
    if all(abs(r - 100.0) < 2 for r in ratio):
        return "marker is 100x too large: units were converted twice"
    if abs(ext[2] - exp[2]) > 0.5:
        return "height is wrong: the up axis was not converted"
    return f"marker bounds unexpected: got {got_min}..{got_max}, expected {exp_min}..{exp_max}"


def main() -> None:
    bundle = bundle_dir()
    with open(os.path.join(bundle, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    tol = float(manifest.get("tolerance_cm", 0.5))
    folder = manifest["mesh_folder"]
    failures: list[str] = []
    report: dict = {"meshes": [], "placements": [], "summary": {}}
    sub = unreal.get_editor_subsystem(unreal.StaticMeshEditorSubsystem)

    meshes = {}
    for m in manifest["meshes"]:
        name = m["name"]
        entry = {"mesh": name}
        report["meshes"].append(entry)
        path = f"{folder}/{name}.{name}"
        asset = unreal.EditorAssetLibrary.load_asset(path) if unreal.EditorAssetLibrary.does_asset_exist(path) else None
        if not isinstance(asset, unreal.StaticMesh):
            failures.append(f"{name}: asset missing at {path}")
            entry["ok"] = False
            continue
        meshes[name] = asset
        box = asset.get_bounding_box()
        got_min, got_max = vec(box.min), vec(box.max)
        exp_min, exp_max = m["local_bounds_cm"]["min"], m["local_bounds_cm"]["max"]
        hulls = sub.get_convex_collision_count(asset)
        try:
            nanite = bool(sub.get_nanite_settings(asset).get_editor_property("enabled"))
        except Exception:
            nanite = bool(asset.get_editor_property("nanite_settings").get_editor_property("enabled"))
        entry.update({"local_min": got_min, "local_max": got_max, "convex_hulls": hulls, "nanite": nanite})
        problems = []
        if not (close(got_min, exp_min, tol) and close(got_max, exp_max, tol)):
            problems.append(f"local bounds {got_min}..{got_max}, expected {exp_min}..{exp_max}")
        want = m["collision"]["convex_hulls"]
        if want and hulls != want:
            problems.append(f"{hulls} convex hulls, expected {want}")
        if nanite != m["nanite"]:
            problems.append(f"Nanite {'on' if nanite else 'off'}, expected {'on' if m['nanite'] else 'off'}")
        if "Marker" in name:
            entry["diagnosis"] = diagnose_marker(got_min, got_max, exp_min, exp_max)
            unreal.log(f"[ArchitectWorks] handedness: {entry['diagnosis']}")
        entry["problems"] = problems
        entry["ok"] = not problems
        failures.extend(f"{name}: {p}" for p in problems)

    actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors()
    by_id = {}
    for a in actors:
        for t in a.tags:
            t = str(t)
            if t.startswith("aw:"):
                by_id[t[3:]] = a
    counts = Counter()
    bmin = [float("inf")] * 3
    bmax = [float("-inf")] * 3
    for p in manifest["placements"]:
        entry = {"id": p["id"]}
        report["placements"].append(entry)
        actor = by_id.get(p["id"])
        if actor is None:
            failures.append(f"{p['id']}: no actor")
            entry["ok"] = False
            continue
        comp = actor.get_component_by_class(unreal.StaticMeshComponent)
        mesh = comp.get_editor_property("static_mesh") if comp else None
        if mesh is None or mesh.get_name() != p["mesh"]:
            failures.append(f"{p['id']}: actor uses {mesh.get_name() if mesh else None}, expected {p['mesh']}")
        counts[p["mesh"]] += 1
        origin, extent = actor.get_actor_bounds(False)
        got_min = [round(origin.x - extent.x, 3), round(origin.y - extent.y, 3), round(origin.z - extent.z, 3)]
        got_max = [round(origin.x + extent.x, 3), round(origin.y + extent.y, 3), round(origin.z + extent.z, 3)]
        exp = p["expected_world_bounds_cm"]
        ok = close(got_min, exp["min"], tol) and close(got_max, exp["max"], tol)
        entry.update({"ok": ok, "world_min": got_min, "world_max": got_max})
        if not ok:
            failures.append(f"{p['id']}: world bounds {got_min}..{got_max}, expected {exp['min']}..{exp['max']}")
        if not p["id"].startswith("aw."):
            bmin = [min(a, b) for a, b in zip(bmin, got_min)]
            bmax = [max(a, b) for a, b in zip(bmax, got_max)]

    expected_counts = Counter(p["mesh"] for p in manifest["placements"])
    for name, n in expected_counts.items():
        if counts[name] != n:
            failures.append(f"{name}: {counts[name]} actors, expected {n}")
    eb = manifest["expected_building_bounds_cm"]
    if not (close(bmin, eb["min"], tol) and close(bmax, eb["max"], tol)):
        failures.append(f"building bounds {bmin}..{bmax}, expected {eb['min']}..{eb['max']}")

    size = [round(b - a, 1) for a, b in zip(eb["min"], eb["max"])]
    report["summary"] = {
        "passed": not failures,
        "failures": failures,
        "building_size_cm": size,
        "meshes_checked": len(report["meshes"]),
        "actors_checked": len(report["placements"]),
    }
    with open(os.path.join(bundle, "check_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    if failures:
        for msg in failures[:40]:
            unreal.log_error(f"[ArchitectWorks] FAIL {msg}")
        unreal.log_error(f"[ArchitectWorks] import check FAILED: {len(failures)} problems; see check_report.json")
    else:
        unreal.log(f"[ArchitectWorks] import check PASSED: {len(report['meshes'])} meshes, {len(report['placements'])} actors, building {size[0]} x {size[1]} x {size[2]} cm")


main()
