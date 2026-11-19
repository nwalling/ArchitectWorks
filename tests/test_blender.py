"""End to end through Blender: build the .blend, export via collection
exporters, re-import every FBX and compare. Skipped when no Blender is
available (no bpy module and no BLENDER binary)."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys

import pytest

from architectworks.cli import BLENDGEN, blender_command
from architectworks.unreal_bundle import write_bundle

HAVE_BLENDER = importlib.util.find_spec("bpy") is not None or bool(os.environ.get("BLENDER") or shutil.which("blender"))
pytestmark = [pytest.mark.blender, pytest.mark.skipif(not HAVE_BLENDER, reason="Blender not available")]


@pytest.fixture(scope="module")
def built(resolved, tmp_path_factory):
    out = tmp_path_factory.mktemp("build")
    (out / "resolved.json").write_text(json.dumps(resolved), encoding="utf-8")
    write_bundle(resolved, out)
    proc = subprocess.run(blender_command(out / "resolved.json", out, []), capture_output=True, text=True, timeout=600)
    report_path = out / "blender_report.json"
    assert report_path.exists(), proc.stdout[-3000:] + proc.stderr[-3000:]
    return out, json.loads(report_path.read_text(encoding="utf-8")), proc


def test_blender_stage_succeeds(built):
    out, report, proc = built
    assert proc.returncode == 0, proc.stdout[-3000:]
    assert report["ok"]
    assert (out / "generated.blend").stat().st_size > 0


def test_collection_exporters_write_every_mesh(built, resolved):
    out, report, _ = built
    assert report["export"]["mode"] == "collection_export_all"
    assert report["export"]["missing"] == []
    for c in resolved["components"]:
        assert (out / "unreal" / "meshes" / f"{c['name']}.fbx").stat().st_size > 0


def test_reimported_fbx_matches_the_resolved_model(built):
    _, report, _ = built
    bad = [c for c in report["verify"]["components"] if not c["ok"]]
    assert not bad, bad
    for c in report["verify"]["components"]:
        assert c["bounds_error_m"] < 1e-4
        assert c["scale"] == [1.0, 1.0, 1.0]
        assert c["ucx"] == c["expected_ucx"]


def test_fbx_header_declares_metres_y_up(built):
    _, report, _ = built
    gs = report["verify"]["fbx_global_settings"]
    assert gs["UnitScaleFactor"] == pytest.approx(100.0)  # 1 FBX unit = 1 m
    assert gs["UpAxis"] == 1 and gs["UpAxisSign"] == 1  # +Y up in the file


def test_bundle_is_complete(built):
    out, report, _ = built
    for name in ("manifest.json", "import_unreal.py", "check_import.py"):
        assert (out / "unreal" / name).exists()
    assert (out / report["preview_glb"]).stat().st_size > 0


def test_blendgen_runs_without_the_package():
    """blendgen must not import architectworks: stock Blender has no pydantic."""
    src = BLENDGEN.read_text(encoding="utf-8")
    assert "architectworks" not in src.replace("ArchitectWorks", "")
