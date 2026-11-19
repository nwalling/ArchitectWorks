"""A stand-in for Unreal's ``unreal`` Python module, just wide enough to run
import_unreal.py and check_import.py outside the editor.

It does not prove anything about Unreal itself. It catches bugs in the
scripts (wrong argument order, a typo in a path, a quaternion passed in the
wrong component order) and checks that check_import.py reports and diagnoses
failures. The importer is simulated: by default it mirrors Y, as the real FBX
importer is expected to; ``mirror=False`` simulates an importer that does not.
"""

from __future__ import annotations

import json
import os
import types


def make(bundle: str, mirror: bool = True, drop_collision: bool = False) -> types.ModuleType:
    u = types.ModuleType("unreal")
    with open(os.path.join(bundle, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    spec = {m["name"]: m for m in manifest["meshes"]}
    assets: dict[str, object] = {}
    actors: list = []
    u.messages = []  # type: ignore[attr-defined]

    u.log = lambda m: u.messages.append(("log", m))
    u.log_warning = lambda m: u.messages.append(("warning", m))
    u.log_error = lambda m: u.messages.append(("error", m))

    class Name(str):
        pass

    class Vector:
        def __init__(self, x=0.0, y=0.0, z=0.0):
            self.x, self.y, self.z = float(x), float(y), float(z)

    class Quat:
        def __init__(self, x=0.0, y=0.0, z=0.0, w=1.0):
            self.x, self.y, self.z, self.w = x, y, z, w

        def rotator(self):
            return Rotator(self)

        def rotate(self, v):
            # standard q v q*, as FQuat::RotateVector
            qx, qy, qz, w = self.x, self.y, self.z, self.w
            tx = 2 * (qy * v[2] - qz * v[1])
            ty = 2 * (qz * v[0] - qx * v[2])
            tz = 2 * (qx * v[1] - qy * v[0])
            return (
                v[0] + w * tx + (qy * tz - qz * ty),
                v[1] + w * ty + (qz * tx - qx * tz),
                v[2] + w * tz + (qx * ty - qy * tx),
            )

    class Rotator:
        def __init__(self, q):
            self.q = q

    class Box:
        def __init__(self, mn, mx):
            self.min, self.max = Vector(*mn), Vector(*mx)

    class Props:
        def __init__(self, **kw):
            self._p = dict(kw)

        def set_editor_property(self, k, v):
            self._p[k] = v

        def get_editor_property(self, k):
            return self._p[k]

    class NaniteSettings(Props):
        pass

    class StaticMesh(Props):
        def __init__(self, name):
            super().__init__(nanite_settings=NaniteSettings(enabled=True))
            m = spec[name]
            lo, hi = list(m["local_bounds_cm"]["min"]), list(m["local_bounds_cm"]["max"])
            if not mirror:
                lo[1], hi[1] = -hi[1], -lo[1]
            self._name, self._lo, self._hi = name, lo, hi
            self._hulls = 0 if drop_collision else m["collision"]["convex_hulls"]

        def get_name(self):
            return self._name

        def get_bounding_box(self):
            return Box(self._lo, self._hi)

    class StaticMeshComponent:
        pass

    class Actor(Props):
        def __init__(self, mesh, loc, rot):
            super().__init__(tags=[])
            self.mesh, self.loc, self.rot = mesh, loc, rot
            self.label = self.folder = None

        @property
        def tags(self):
            return self._p["tags"]

        def set_actor_label(self, s):
            self.label = s

        def set_folder_path(self, s):
            self.folder = s

        def get_component_by_class(self, cls):
            comp = Props(static_mesh=self.mesh)
            return comp

        def get_actor_bounds(self, only_colliding, *rest):
            lo, hi = self.mesh._lo, self.mesh._hi
            pts = []
            for x in (lo[0], hi[0]):
                for y in (lo[1], hi[1]):
                    for z in (lo[2], hi[2]):
                        r = self.rot.q.rotate((x, y, z))
                        pts.append((r[0] + self.loc.x, r[1] + self.loc.y, r[2] + self.loc.z))
            mn = [min(p[i] for p in pts) for i in range(3)]
            mx = [max(p[i] for p in pts) for i in range(3)]
            return Vector(*[(a + b) / 2 for a, b in zip(mn, mx)]), Vector(*[(b - a) / 2 for a, b in zip(mn, mx)])

    class AssetImportTask(Props):
        def __init__(self):
            super().__init__(imported_object_paths=[], options=None)

    class AssetTools:
        def import_asset_tasks(self, tasks):
            for t in tasks:
                name = os.path.splitext(os.path.basename(t.get_editor_property("filename")))[0]
                assert os.path.isfile(t.get_editor_property("filename")), t.get_editor_property("filename")
                path = f"{t.get_editor_property('destination_path')}/{name}.{name}"
                assets[path] = StaticMesh(name)
                t.set_editor_property("imported_object_paths", [path])

    class AssetToolsHelpers:
        @staticmethod
        def get_asset_tools():
            return AssetTools()

    class EditorAssetLibrary:
        @staticmethod
        def does_asset_exist(path):
            return path in assets

        @staticmethod
        def load_asset(path):
            return assets.get(path)

        @staticmethod
        def list_assets(folder, recursive=True, include_folder=False):
            return [p for p in assets if p.startswith(folder)]

        @staticmethod
        def save_loaded_asset(asset):
            return True

    class EditorActorSubsystem:
        def get_all_level_actors(self):
            return list(actors)

        def destroy_actor(self, a):
            actors.remove(a)

        def spawn_actor_from_object(self, mesh, loc, rot):
            a = Actor(mesh, loc, rot)
            actors.append(a)
            return a

    class StaticMeshEditorSubsystem:
        def get_convex_collision_count(self, mesh):
            return mesh._hulls

        def get_nanite_settings(self, mesh):
            return mesh.get_editor_property("nanite_settings")

        def set_nanite_settings(self, mesh, settings, apply):
            mesh.set_editor_property("nanite_settings", settings)

    class LevelEditorSubsystem:
        def save_current_level(self):
            return True

    class InterchangeGenericAssetsPipeline(Props):
        def __init__(self):
            super().__init__(mesh_pipeline=Props())

    class InterchangePipelineStackOverride:
        def __init__(self):
            self.pipelines = []

        def add_pipeline(self, p):
            self.pipelines.append(p)

    class ScopedSlowTask:
        def __init__(self, *a):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def make_dialog(self, *a):
            pass

        def enter_progress_frame(self, *a):
            pass

    subsystems = {EditorActorSubsystem: EditorActorSubsystem(), StaticMeshEditorSubsystem: StaticMeshEditorSubsystem(), LevelEditorSubsystem: LevelEditorSubsystem()}
    for obj in (
        Name, Vector, Quat, Rotator, Box, StaticMesh, StaticMeshComponent, AssetImportTask, AssetToolsHelpers, EditorAssetLibrary,
        EditorActorSubsystem, StaticMeshEditorSubsystem, LevelEditorSubsystem, InterchangeGenericAssetsPipeline,
        InterchangePipelineStackOverride, ScopedSlowTask,
    ):
        setattr(u, obj.__name__, obj)
    u.get_editor_subsystem = lambda cls: subsystems[cls]
    u.actors = actors  # type: ignore[attr-defined]
    return u
