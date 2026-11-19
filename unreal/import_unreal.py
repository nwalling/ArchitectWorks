"""Import an ArchitectWorks bundle into the open Unreal Engine level.

Copied into every bundle (out/<spec>/unreal/) next to manifest.json and the
meshes folder. Run it inside the Unreal Editor with the Python Editor Script
Plugin enabled:

    In the editor:  Output Log > Cmd > py "C:/path/to/unreal/import_unreal.py"
    Headless:       UnrealEditor-Cmd.exe MyProject.uproject
                        -ExecutePythonScript="C:/path/to/unreal/import_unreal.py"
                        -unattended -nosplash -nullrhi

The bundle folder defaults to this script's folder. Pass another as the
first argument, or set AW_BUNDLE. Add --save-level to save the level after
spawning. Re-running replaces the actors from the previous run (they are
found by tag) and re-imports the meshes.

Written against Unreal Engine 5.8 and not run in the environment that
generated it: every API call that might differ between engine versions is
wrapped, and anything that did not apply is listed in import_log.json. Run
check_import.py afterwards; it is the real test.
"""

from __future__ import annotations

import json
import os
import sys
import time

import unreal

LOG: dict = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "pipeline": {}, "imports": [], "nanite": [], "spawned": 0, "warnings": []}


def log(msg: str) -> None:
    unreal.log(f"[ArchitectWorks] {msg}")


def warn(msg: str) -> None:
    LOG["warnings"].append(msg)
    unreal.log_warning(f"[ArchitectWorks] {msg}")


def script_dir() -> str:
    f = globals().get("__file__") or (sys.argv[0] if sys.argv and sys.argv[0].endswith(".py") else None)
    return os.path.dirname(os.path.abspath(f)) if f else os.getcwd()


def parse_args() -> tuple[str, bool]:
    args = [a for a in sys.argv[1:] if not a.endswith(".py")]
    save = "--save-level" in args
    paths = [a for a in args if not a.startswith("--")]
    if paths:
        return os.path.abspath(paths[0]), save
    if os.environ.get("AW_BUNDLE"):
        return os.path.abspath(os.environ["AW_BUNDLE"]), save
    return script_dir(), save


# ---- import -----------------------------------------------------------------


def _set(obj, name: str, value, applied: list, missed: list) -> None:
    try:
        obj.set_editor_property(name, value)
        applied.append(name)
    except Exception:  # property names move between engine versions
        missed.append(name)


def make_pipeline_options():
    """Interchange options: collision from UCX_ names, one hull per UCX, no
    mesh combining. Returns an object for AssetImportTask.options, or None to
    use the project's defaults."""
    applied: list = []
    missed: list = []
    try:
        pipeline = unreal.InterchangeGenericAssetsPipeline()
    except Exception as err:
        warn(f"InterchangeGenericAssetsPipeline unavailable ({err}); using the project's import defaults")
        return None
    mesh = None
    try:
        mesh = pipeline.get_editor_property("mesh_pipeline")
    except Exception:
        missed.append("mesh_pipeline")
    if mesh is not None:
        _set(mesh, "import_static_meshes", True, applied, missed)
        _set(mesh, "import_skeletal_meshes", False, applied, missed)
        _set(mesh, "combine_static_meshes", False, applied, missed)
        _set(mesh, "import_collision", True, applied, missed)
        _set(mesh, "import_collision_according_to_mesh_name", True, applied, missed)
        _set(mesh, "one_convex_hull_per_ucx", True, applied, missed)
    options = None
    try:
        options = unreal.InterchangePipelineStackOverride()
        options.add_pipeline(pipeline)
    except Exception as err:
        warn(f"InterchangePipelineStackOverride unavailable ({err}); passing the pipeline directly")
        options = pipeline
    LOG["pipeline"] = {"applied": applied, "not_found": missed, "options_type": type(options).__name__}
    if missed:
        warn(f"pipeline properties not found in this engine version: {', '.join(missed)}")
    return options


def import_meshes(bundle: str, manifest: dict) -> dict:
    tools = unreal.AssetToolsHelpers.get_asset_tools()
    folder = manifest["mesh_folder"]
    options = make_pipeline_options()
    tasks = []
    for m in manifest["meshes"]:
        task = unreal.AssetImportTask()
        task.set_editor_property("filename", os.path.join(bundle, m["file"]))
        task.set_editor_property("destination_path", folder)
        task.set_editor_property("automated", True)
        task.set_editor_property("replace_existing", True)
        task.set_editor_property("save", True)
        if options is not None:
            try:
                task.set_editor_property("options", options)
            except Exception as err:
                warn(f"could not attach import options: {err}")
                options = None
        tasks.append((m, task))
    log(f"importing {len(tasks)} meshes into {folder}")
    tools.import_asset_tasks([t for _, t in tasks])
    meshes = {}
    for m, task in tasks:
        paths = [str(p) for p in task.get_editor_property("imported_object_paths")]
        asset = find_mesh(folder, m["name"])
        LOG["imports"].append({"mesh": m["name"], "imported": paths, "found": asset is not None})
        if asset is None:
            warn(f"{m['name']}: no StaticMesh asset after import (imported: {paths})")
            continue
        meshes[m["name"]] = asset
        set_nanite(asset, m["nanite"])
    return meshes


def find_mesh(folder: str, name: str):
    for path in (f"{folder}/{name}.{name}", f"{folder}/{name}"):
        if unreal.EditorAssetLibrary.does_asset_exist(path):
            asset = unreal.EditorAssetLibrary.load_asset(path)
            if isinstance(asset, unreal.StaticMesh):
                return asset
    for path in unreal.EditorAssetLibrary.list_assets(folder, recursive=True, include_folder=False):
        if path.split("/")[-1].split(".")[0] == name:
            asset = unreal.EditorAssetLibrary.load_asset(path)
            if isinstance(asset, unreal.StaticMesh):
                return asset
    return None


def set_nanite(mesh, enabled: bool) -> None:
    try:
        sub = unreal.get_editor_subsystem(unreal.StaticMeshEditorSubsystem)
        settings = sub.get_nanite_settings(mesh)
        if settings.get_editor_property("enabled") != enabled:
            settings.set_editor_property("enabled", enabled)
            sub.set_nanite_settings(mesh, settings, True)
        LOG["nanite"].append({"mesh": mesh.get_name(), "enabled": enabled})
    except Exception as err:
        try:
            settings = mesh.get_editor_property("nanite_settings")
            settings.set_editor_property("enabled", enabled)
            mesh.set_editor_property("nanite_settings", settings)
            LOG["nanite"].append({"mesh": mesh.get_name(), "enabled": enabled, "via": "property"})
        except Exception as err2:
            warn(f"{mesh.get_name()}: could not set Nanite ({err}; {err2})")
    unreal.EditorAssetLibrary.save_loaded_asset(mesh)


# ---- placement ----------------------------------------------------------------


def spawn(manifest: dict, meshes: dict) -> int:
    actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    tag = manifest["tag"]
    removed = 0
    for actor in actors.get_all_level_actors():
        if tag in [str(t) for t in actor.tags]:
            actors.destroy_actor(actor)
            removed += 1
    if removed:
        log(f"removed {removed} actors from a previous run")
    count = 0
    for p in manifest["placements"]:
        mesh = meshes.get(p["mesh"])
        if mesh is None:
            warn(f"{p['id']}: mesh {p['mesh']} was not imported; skipped")
            continue
        loc = unreal.Vector(*p["location_cm"])
        qx, qy, qz, qw = p["rotation_quat_xyzw"]
        rot = unreal.Quat(x=qx, y=qy, z=qz, w=qw).rotator()
        actor = actors.spawn_actor_from_object(mesh, loc, rot)
        if actor is None:
            warn(f"{p['id']}: spawn failed")
            continue
        actor.set_actor_label(p["id"])
        actor.set_folder_path(p["folder"])
        actor.set_editor_property("tags", [unreal.Name(tag), unreal.Name("aw:" + p["id"])])
        count += 1
    return count


def main() -> None:
    bundle, save = parse_args()
    with open(os.path.join(bundle, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    log(f"bundle {bundle}: {len(manifest['meshes'])} meshes, {len(manifest['placements'])} placements")
    with unreal.ScopedSlowTask(3, "ArchitectWorks import") as task:
        task.make_dialog(False)
        task.enter_progress_frame(1, "Importing meshes")
        meshes = import_meshes(bundle, manifest)
        task.enter_progress_frame(1, "Placing actors")
        LOG["spawned"] = spawn(manifest, meshes)
        task.enter_progress_frame(1, "Saving")
        if save:
            unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).save_current_level()
    LOG["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(bundle, "import_log.json"), "w", encoding="utf-8") as f:
        json.dump(LOG, f, indent=1)
    log(f"imported {len(meshes)}/{len(manifest['meshes'])} meshes, spawned {LOG['spawned']} actors, {len(LOG['warnings'])} warnings")
    log("next: run check_import.py from the same folder")


main()
