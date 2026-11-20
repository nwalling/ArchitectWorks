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

import yaml

from architectworks.resolve import resolve
from architectworks.spec import load_spec
from architectworks.spec.model import json_schema
from architectworks.unreal_bundle import REPO_ROOT, write_bundle
from architectworks.units import to_metres

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
        if "views" in (yaml.safe_load(calib_path.read_text(encoding="utf-8")) or {}):
            datum = spec.levels[spec.datum].elev
            levels_m = {k: to_metres(lv.elev - datum) for k, lv in spec.levels.items()}
            fits = sheets.fit_elevations(calib_path, resolved["points"], levels_m, float(datum) / 12.0)
            fits["calibration"] = str(calib_path.resolve())
            (out / "sheets").mkdir(parents=True, exist_ok=True)
            (out / "sheets" / f"{sid}.elevation-fit.json").write_text(json.dumps(fits, indent=1) + "\n", encoding="utf-8")
            for vname, v in fits["views"].items():
                worst = max([abs(e) for e in v["residual_px"]["across"].values()] + [abs(e) for e in v["residual_px"]["levels"].values()])
                print(f"  sheet {sid}, {vname} elevation (looking {v['looking']}): {v['px_per_ft'][0]} px/ft across, {v['px_per_ft'][1]} up; worst residual {worst:.1f} px")
            reports.append({"sheet": sid, "elevations": fits})
            continue
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


def elevation_fits(out: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted((out / "sheets").glob("*.elevation-fit.json"))]


def cmd_compare_elevations(out: Path, report: dict) -> list[dict]:
    """Lay each drawn elevation over the matching orthographic render."""
    from architectworks import sheets

    renders = {r["looking"]: r for r in report.get("elevations", [])}
    results = []
    for fits in elevation_fits(out):
        drawing = sheets.load_elevation_sheet(Path(fits["calibration"]))
        for vname, view in fits["views"].items():
            r = renders.get(view["looking"])
            if r is None:
                continue
            dest = out / "sheets" / f"{Path(fits['sheet']).stem}.{vname}.compare.jpg"
            res = sheets.compare_elevation(view, r, out, drawing, dest, f"{vname.capitalize()} elevation")
            results.append(res)
            print(f"  {vname} elevation: {res['drawing_ink_inside_model']:.0%} of the drawing's ink falls inside the model's silhouette; {dest}")
    return results


def cmd_blender(out: Path, preview: bool, samples: int) -> dict:
    extra = ["--preview", "--samples", str(samples)] if preview else []
    looking = sorted({v["looking"] for f in elevation_fits(out) for v in f["views"].values()})
    if preview and looking:
        extra += ["--elevations", ",".join(looking)]
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
    if report.get("elevations"):
        try:
            report["compare"] = cmd_compare_elevations(out, report)
        except ImportError as err:
            print(f"  skipping elevation comparisons: {err}")
    if proc.returncode != 0 or not report["ok"]:
        raise SystemExit(f"Blender stage failed; see {report_path} and {log}")
    return report


# ---- phase 2: reading sheets ----

def reader_paths(sheet: Path) -> dict[str, Path]:
    """projects/<p>/sheets/<s>.jpg -> where the reader's files go."""
    project = sheet.resolve().parent.parent
    return {
        "readings": project / "readings",
        "recordings": project / "readings" / "recordings" / sheet.stem,
        "out": project / "out" / "reader",
    }


def _load_sheet(sheet: Path):
    try:
        from architectworks.reader.detect import detect, load_flat
    except ImportError as err:
        raise SystemExit(f"reading sheets needs the sheets extra (pip install -e .[sheets]): {err}")
    flat = load_flat(sheet)
    return flat, detect(flat)


def cmd_detect(sheet: Path, out: Path | None) -> None:
    from architectworks.reader.views import render

    flat, det = _load_sheet(sheet)
    out = out or reader_paths(sheet)["out"]
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{sheet.stem}.marks.json").write_text(json.dumps(det.to_json(), indent=1) + "\n", encoding="utf-8")
    h, w = flat.shape
    (out / f"{sheet.stem}.marks.jpg").write_bytes(render(flat, 0, 0, w, h, det, max_side=3000, max_zoom=1.0).image)
    kinds = Counter(m.kind for m in det.marks)
    print(f"{sheet}: {kinds['blob']} blobs (M), {kinds['convergence']} convergences (C), {len(det.segments)} segments")
    print(f"  {out / (sheet.stem + '.marks.json')}, {out / (sheet.stem + '.marks.jpg')}")


def cmd_read(sheet: Path, out: Path | None, offline: bool, max_turns: int) -> None:
    from architectworks.reader import claude

    flat, det = _load_sheet(sheet)
    paths = reader_paths(sheet)
    out = out or paths["readings"] / f"{sheet.stem}.claude.json"
    if not offline and not os.environ.get("ANTHROPIC_API_KEY") and not any((paths["recordings"]).glob("*.json")):
        print("note: ANTHROPIC_API_KEY is not set and there are no recordings for this sheet; the API call will fail.")
    print(f"reading {sheet} with {claude.MODEL} (effort {claude.EFFORT}); recordings in {paths['recordings']}")
    try:
        result = claude.read_plan(flat, det, recordings=paths["recordings"], offline=offline, max_turns=max_turns)
    except claude.ReaderError as err:
        raise SystemExit(f"reader failed: {err}")
    except Exception as err:
        import anthropic

        if isinstance(err, anthropic.APIError) or "authentication method" in str(err):
            raise SystemExit(f"reader could not use the API: {err}\nSet ANTHROPIC_API_KEY, or use --offline to replay recordings.")
        raise
    data = {"_run": {"model": claude.MODEL, "effort": claude.EFFORT, **result.summary()}, **result.reading.model_dump()}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    r = result.reading
    print(f"  {result.turns} turns, {result.views} views; {result.called} API calls, {result.replayed} replayed; served by {', '.join(result.served_by) or '-'}")
    print(f"  usage: {result.usage}")
    print(f"  read {len(r.center_points)} center points, {len(r.columns)} columns, {len(r.chains)} chains, {len(r.notes)} notes, {len(r.unreadable)} unreadable -> {out}")


def cmd_draft(reading_path: Path, sheet: Path, out: Path | None, key: Path | None) -> None:
    from architectworks.reader.draft import build_draft
    from architectworks.reader.schema import load_reading

    flat, det = _load_sheet(sheet)
    draft = build_draft(load_reading(reading_path), flat, det).to_json()
    out = out or reader_paths(sheet)["out"] / f"{reading_path.stem}.draft.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(draft, indent=1) + "\n", encoding="utf-8")
    f = draft["fit"]
    print(f"draft from {reading_path}: {len(draft['control'])} control points, {len(draft['pavilions'])} pavilions -> {out}")
    print(f"  fit: {f.get('dimensioned_both_axes')} features dimensioned on both axes, residual {f.get('feature_residual_in')}")
    for note in draft["review"]:
        print(f"  [review] {note}")
    if key:
        cmd_score(out, key, sheet)


def cmd_score(draft_path: Path, key_path: Path, image: Path | None = None) -> dict:
    from architectworks.reader.score import score, sheet_for_image

    key = load_spec(key_path)
    sheet = sheet_for_image(key, image) if image else None
    result = score(json.loads(draft_path.read_text(encoding="utf-8")), key, sheet)
    if sheet:
        print(f"scoring the key's points from sheet {sheet!r} only")
    out = draft_path.with_name(draft_path.name.replace(".draft.json", "") + ".score.json")
    out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    s = result["summary"]
    cp, pv = s["center_points"], s["pavilions"]
    print(f"score against {key_path}: {s['right']}/{s['items']} ({s['score']:.0%})")
    print(f"  center points: {cp['matched']}/{cp['key']} found, {cp['exact']} exact, {cp['within_2_in']} within 2 in; missed {cp['missed'] or 'none'}")
    print(f"  pavilions: {pv['all_checks_pass']}/{pv['key']} fully right; missed {pv['missed'] or 'none'}")
    for p in result["points"]:
        if p.get("key") and not p["exact"]:
            print(f"    {p['key']}: off by {p['error_in']} in ({p['dx_in']:+}, {p['dy_in']:+}); draft basis {p['draft_basis']}, key basis {p['key_basis']}")
    for p in result["pavilions"]:
        bad = [k for k, ok in p.get("checks", {}).items() if not ok]
        if bad:
            print(f"    {p['key']}: wrong {', '.join(bad)}")
    print(f"  -> {out}")
    return result


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

    p = sub.add_parser("detect", help="find candidate marks on a sheet photo")
    p.add_argument("sheet", type=Path)
    p.add_argument("-o", "--out", type=Path, help="directory (default projects/<p>/out/reader)")

    p = sub.add_parser("read", help="have Claude read a plan sheet (needs ANTHROPIC_API_KEY unless replaying)")
    p.add_argument("sheet", type=Path)
    p.add_argument("-o", "--out", type=Path, help="reading file (default projects/<p>/readings/<sheet>.claude.json)")
    p.add_argument("--offline", action="store_true", help="replay recordings only; never call the API")
    p.add_argument("--max-turns", type=int, default=40)

    p = sub.add_parser("draft", help="turn a reading into a draft spec")
    p.add_argument("reading", type=Path)
    p.add_argument("sheet", type=Path)
    p.add_argument("-o", "--out", type=Path)
    p.add_argument("--score", type=Path, metavar="SPEC", help="also score the draft against this spec")

    p = sub.add_parser("score", help="score a draft against a hand-written spec")
    p.add_argument("draft", type=Path)
    p.add_argument("spec", type=Path)
    p.add_argument("--sheet", type=Path, help="the sheet image the draft was read from: score only the key's points from that sheet")

    args = ap.parse_args(argv)
    if args.cmd == "detect":
        cmd_detect(args.sheet, args.out)
        return 0
    if args.cmd == "read":
        cmd_read(args.sheet, args.out, args.offline, args.max_turns)
        return 0
    if args.cmd == "draft":
        cmd_draft(args.reading, args.sheet, args.out, args.score)
        return 0
    if args.cmd == "score":
        cmd_score(args.draft, args.spec, args.sheet)
        return 0
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
