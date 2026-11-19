"""Command line: ``aw build projects/example-house/pavilion_a.yaml``."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

from architectworks.resolve import resolve
from architectworks.spec import load_spec
from architectworks.spec.model import json_schema
from architectworks.unreal_bundle import REPO_ROOT, write_bundle

BLENDGEN = REPO_ROOT / "blendgen" / "build.py"


def default_out(spec_path: Path) -> Path:
    return spec_path.parent / "out" / spec_path.stem


def cmd_sheets(spec_path: Path, spec, resolved: dict, out: Path) -> list[dict]:
    """Fit every calibrated sheet, check the model against it, and hand the
    straightened sheet to Blender as a reference plane."""
    reports = []
    for sid, sheet in spec.sheets.items():
        if not sheet.calibration:
            continue
        try:
            from architectworks import sheets
        except ImportError as err:
            print(f"  skipping sheet checks: {err}")
            return reports
        calib_path = spec_path.parent / sheet.calibration
        calib = sheets.fit(calib_path, resolved["points"])
        report = sheets.overlay(resolved, calib, calib_path.parent / calib["sheet"], out / "sheets")
        reports.append(report)
        f = report["fit"]
        print(f"  sheet {sid}: fitted on {f['points_used']} points, residual {f['residual_in']['mean']}\" mean, {f['residual_in']['max']}\" max ({f['residual_in']['worst']})")
        print(f"    model on ink: " + ", ".join(f"{k} {v:.0%}" for k, v in report["on_ink_by_kind"].items()) + f"; overlay {out / 'sheets' / (Path(calib['sheet']).stem + '.overlay.jpg')}")
        x0, y0, x1, y1 = report["plan_bounds_m"]
        resolved.setdefault("references", []).append(
            {"sheet": sid, "image": str((out / "sheets" / (Path(calib["sheet"]).stem + ".rectified.png")).resolve()), "min": [x0, y0], "max": [x1, y1], "z": -0.15}
        )
    return reports


def cmd_resolve(spec_path: Path, out: Path, marker: bool = True, sheets: bool = True) -> dict:
    spec = load_spec(spec_path)
    resolved = resolve(spec, marker=marker)
    out.mkdir(parents=True, exist_ok=True)
    if sheets:
        cmd_sheets(spec_path, spec, resolved, out)
    (out / "resolved.json").write_text(json.dumps(resolved, separators=(",", ":"), sort_keys=False) + "\n", encoding="utf-8")
    counts = Counter(c["status"] for c in resolved["checks"])
    print(f"resolved {spec_path} -> {out / 'resolved.json'}")
    print(f"  {len(resolved['components'])} components, {len(resolved['placements'])} placements")
    print(f"  checks: {counts.get('pass', 0)} pass, {counts.get('conflict', 0)} conflict, {counts.get('info', 0)} info")
    for c in resolved["checks"]:
        if c["status"] != "pass":
            print(f"    [{c['status']}] {c['message']}")
    return resolved


def blender_command(resolved_path: Path, out: Path, extra: list[str]) -> list[str]:
    if importlib.util.find_spec("bpy") is not None:
        return [sys.executable, str(BLENDGEN), str(resolved_path), str(out), *extra]
    blender = os.environ.get("BLENDER") or shutil.which("blender")
    if not blender:
        raise SystemExit("No Blender found: install the bpy wheel (pip install -e .[blender]) or set BLENDER to a Blender 5.2 binary.")
    return [blender, "--background", "--factory-startup", "--python", str(BLENDGEN), "--", str(resolved_path), str(out), *extra]


def cmd_blender(out: Path, preview: bool, samples: int) -> dict:
    extra = ["--preview", "--samples", str(samples)] if preview else []
    cmd = blender_command(out / "resolved.json", out, extra)
    print("running:", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    log = out / "blender.log"
    log.write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr, encoding="utf-8")
    report_path = out / "blender_report.json"
    if not report_path.exists():
        print(proc.stdout[-4000:], proc.stderr[-4000:], sep="\n")
        raise SystemExit(f"Blender did not write a report (exit {proc.returncode}); see {log}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    ex = report["export"]
    print(f"  Blender {report['blender']}: {len(ex['files'])} FBX files via {ex['mode']}, {len(ex['missing'])} missing")
    v = report.get("verify")
    if v:
        bad = [c for c in v["components"] if not c["ok"]]
        print(f"  re-import check: {len(v['components']) - len(bad)}/{len(v['components'])} meshes match")
        for c in bad:
            print(f"    {c['component']}: {c.get('error') or ', '.join(c['problems'])}")
        if v.get("fbx_global_settings"):
            print(f"  FBX header: {v['fbx_global_settings']}")
    if report.get("preview_png"):
        print(f"  preview: {out / report['preview_png']}")
    if proc.returncode != 0 or not report["ok"]:
        raise SystemExit(f"Blender stage failed; see {report_path} and {log}")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="aw", description="ArchitectWorks: Building Spec -> Blender -> Unreal")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("resolve", help="validate a spec and resolve it to exact geometry")
    p.add_argument("spec", type=Path)
    p.add_argument("-o", "--out", type=Path)
    p.add_argument("--no-marker", action="store_true")
    p.add_argument("--no-sheets", action="store_true", help="skip fitting and checking calibrated sheets")

    p = sub.add_parser("build", help="resolve, write the Unreal bundle, build the .blend and export")
    p.add_argument("spec", type=Path)
    p.add_argument("-o", "--out", type=Path)
    p.add_argument("--preview", action="store_true", help="also render a preview image (Cycles, CPU)")
    p.add_argument("--samples", type=int, default=48)
    p.add_argument("--no-blender", action="store_true", help="stop after the manifest")
    p.add_argument("--no-marker", action="store_true")
    p.add_argument("--no-sheets", action="store_true", help="skip fitting and checking calibrated sheets")

    p = sub.add_parser("schema", help="print the Building Spec JSON Schema")
    p.add_argument("-o", "--out", type=Path)

    args = ap.parse_args(argv)
    if args.cmd == "schema":
        text = json.dumps(json_schema(), indent=1) + "\n"
        if args.out:
            args.out.write_text(text, encoding="utf-8")
        else:
            sys.stdout.write(text)
        return 0

    out = args.out or default_out(args.spec)
    resolved = cmd_resolve(args.spec, out, marker=not args.no_marker, sheets=not args.no_sheets)
    if args.cmd == "resolve":
        return 0
    manifest = write_bundle(resolved, out)
    print(f"  Unreal manifest: {manifest}")
    if not args.no_blender:
        cmd_blender(out, args.preview, args.samples)
        print(f"done: {out / 'generated.blend'} and {out / 'unreal'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
