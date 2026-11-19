"""Run the Unreal scripts against a stand-in ``unreal`` module.

This exercises the scripts' own logic (argument handling, quaternion order,
tags, bounds arithmetic, diagnosis). The real test of Unreal's importer is
running check_import.py inside the editor.
"""

import json
import runpy
import sys
from pathlib import Path

import pytest

from architectworks.unreal_bundle import write_bundle
from tests import fake_unreal


def make_bundle(resolved, tmp_path: Path) -> Path:
    write_bundle(resolved, tmp_path)
    bundle = tmp_path / "unreal"
    (bundle / "meshes").mkdir(exist_ok=True)
    for m in json.loads((bundle / "manifest.json").read_text())["meshes"]:
        (bundle / m["file"]).write_bytes(b"stand-in")
    return bundle


def run(script: Path, bundle: Path, fake, monkeypatch):
    monkeypatch.setitem(sys.modules, "unreal", fake)
    monkeypatch.setattr(sys, "argv", [str(script), str(bundle)])
    runpy.run_path(str(script), run_name="__main__")


@pytest.mark.parametrize("mirror, drop, expect_pass", [(True, False, True), (False, False, False), (True, True, False)])
def test_import_then_check(resolved, tmp_path, monkeypatch, mirror, drop, expect_pass):
    bundle = make_bundle(resolved, tmp_path)
    fake = fake_unreal.make(str(bundle), mirror=mirror, drop_collision=drop)
    run(bundle / "import_unreal.py", bundle, fake, monkeypatch)
    log = json.loads((bundle / "import_log.json").read_text())
    assert log["spawned"] == len(resolved["placements"])
    run(bundle / "check_import.py", bundle, fake, monkeypatch)
    report = json.loads((bundle / "check_report.json").read_text())
    assert report["summary"]["passed"] is expect_pass, report["summary"]["failures"][:5]
    marker = next(m for m in report["meshes"] if "Marker" in m["mesh"])
    if mirror:
        assert marker["diagnosis"].startswith("marker matches")
    else:
        assert "not mirrored" in marker["diagnosis"]
    if drop:
        assert any("convex hulls" in f for f in report["summary"]["failures"])


def test_rerun_replaces_previous_actors(resolved, tmp_path, monkeypatch):
    bundle = make_bundle(resolved, tmp_path)
    fake = fake_unreal.make(str(bundle))
    run(bundle / "import_unreal.py", bundle, fake, monkeypatch)
    run(bundle / "import_unreal.py", bundle, fake, monkeypatch)
    assert len(fake.actors) == len(resolved["placements"])
