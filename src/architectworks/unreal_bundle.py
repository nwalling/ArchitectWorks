"""The Unreal bundle: a manifest of meshes and placements in Unreal's own
space (centimetres, left-handed), plus the scripts that import and check it.

Every expected number Unreal will be tested against is computed here, in
plain Python, from the same resolved model that Blender built from.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from architectworks.geom.xform import (
    transformed_box_bounds,
    ue_bounds_cm,
    ue_point_cm,
    ue_quat_xyzw,
    ue_rotator_from_quat,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
UNREAL_SCRIPTS = ("import_unreal.py", "check_import.py")


def _r(v, n: int = 4) -> list[float]:
    return [round(x, n) + 0.0 for x in v]


def _camel(text: str) -> str:
    return "".join(p[:1].upper() + p[1:] for p in text.replace("_", "-").split("-") if p)


def build_manifest(resolved: dict[str, Any]) -> dict[str, Any]:
    comps = {c["name"]: c for c in resolved["components"]}
    building = resolved["building"]
    root = f"/Game/ArchitectWorks/{_camel(building['id'])}"
    meshes = []
    for c in resolved["components"]:
        lmin, lmax = ue_bounds_cm(tuple(c["bounds"]["min"]), tuple(c["bounds"]["max"]))
        meshes.append(
            {
                "name": c["name"],
                "file": f"meshes/{c['name']}.fbx",
                "category": c["category"],
                "material_slots": c["material_slots"],
                "nanite": c["nanite"],
                "collision": {"convex_hulls": len(c["collision"])},
                "local_bounds_cm": {"min": _r(lmin), "max": _r(lmax)},
                "triangles": c["triangles"],
            }
        )
    placements = []
    bmin_all = [float("inf")] * 3
    bmax_all = [float("-inf")] * 3
    for p in resolved["placements"]:
        m = tuple(tuple(row) for row in p["matrix"])
        c = comps[p["component"]]
        loc = ue_point_cm((m[0][3], m[1][3], m[2][3]))
        q = ue_quat_xyzw(m)
        wmin_b, wmax_b = transformed_box_bounds(m, tuple(c["bounds"]["min"]), tuple(c["bounds"]["max"]))
        wmin, wmax = ue_bounds_cm(wmin_b, wmax_b)
        if p["category"] != "Fixtures":
            bmin_all = [min(a, b) for a, b in zip(bmin_all, wmin)]
            bmax_all = [max(a, b) for a, b in zip(bmax_all, wmax)]
        placements.append(
            {
                "id": p["id"],
                "mesh": p["component"],
                "folder": f"ArchitectWorks/{building['id']}/{p['category']}",
                "location_cm": _r(loc),
                "rotation_quat_xyzw": _r(q, 9),
                "rotation_rotator_deg": {k: round(v, 4) + 0.0 for k, v in ue_rotator_from_quat(q).items()},
                "scale": [1.0, 1.0, 1.0],
                "expected_world_bounds_cm": {"min": _r(wmin), "max": _r(wmax)},
                "basis": p["basis"],
                "confidence": p["confidence"],
                "element": p["element"],
            }
        )
    return {
        "format": "architectworks.unreal-manifest",
        "version": 1,
        "generator": resolved["generator"],
        "building": building,
        "space": {"units": "centimetres", "axes": "Unreal: X forward, Y right, Z up (left-handed)", "from_blender": "(x, y, z) m -> (100x, -100y, 100z) cm"},
        "content_root": root,
        "mesh_folder": f"{root}/Meshes",
        "tag": f"AW_{building['code']}",
        "import_settings": {
            "collision_from_mesh_names": True,
            "one_convex_hull_per_ucx": True,
            "combine_meshes": False,
            "nanite_for_opaque": True,
        },
        "meshes": meshes,
        "placements": placements,
        "expected_building_bounds_cm": {"min": _r(bmin_all), "max": _r(bmax_all)},
        "tolerance_cm": 0.5,
        "checks": [c for c in resolved["checks"] if c["status"] != "pass"],
    }


def write_bundle(resolved: dict[str, Any], out_dir: Path) -> Path:
    import json

    udir = Path(out_dir) / "unreal"
    udir.mkdir(parents=True, exist_ok=True)
    path = udir / "manifest.json"
    path.write_text(json.dumps(build_manifest(resolved), indent=1) + "\n", encoding="utf-8")
    for name in UNREAL_SCRIPTS:
        shutil.copyfile(REPO_ROOT / "unreal" / name, udir / name)
    return path
