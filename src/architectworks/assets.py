"""Textures and plant models from Poly Haven (CC0), fetched on demand.

Asset files are large, so they are not committed: ``aw assets`` downloads
what the spec names into ``assets/polyhaven`` (git-ignored), checks every
file against the md5 Poly Haven publishes, and writes ``index.json`` there
for the Blender stage to read. Anything missing falls back to the plain or
procedural materials.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any

from architectworks.spec.model import BuildingSpec

API = "https://api.polyhaven.com"
USER_AGENT = "ArchitectWorks-asset-fetch/0.1"
TEXTURE_MAPS = {"Diffuse": "diff", "nor_gl": "nor", "Rough": "rough"}
MODEL_RESOLUTION = "1k"


def wanted(spec: BuildingSpec) -> dict[str, list]:
    textures = sorted({(m.texture.polyhaven, m.texture.resolution) for m in spec.types.materials.values() if m.texture})
    models = sorted({p.polyhaven for p in spec.site.planting}) if spec.site else []
    return {"textures": textures, "models": models}


def _get_json(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, md5: str, dest: Path, log) -> int:
    if dest.exists() and _md5(dest) == md5:
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=300) as r, tmp.open("wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    got = _md5(tmp)
    if got != md5:
        tmp.unlink()
        raise IOError(f"{url}: md5 {got} does not match the published {md5}")
    tmp.replace(dest)
    log(f"    {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest.stat().st_size


def fetch(spec: BuildingSpec, root: Path, log=print) -> dict[str, Any]:
    """Download everything the spec names; return (and write) the index."""
    want = wanted(spec)
    index: dict[str, Any] = {"source": "polyhaven.com (CC0)", "textures": {}, "models": {}}
    total = 0
    for asset, res in want["textures"]:
        info = _get_json(f"{API}/info/{asset}")
        files = _get_json(f"{API}/files/{asset}")
        entry = {"dimensions_mm": info.get("dimensions"), "resolution": res, "maps": {}}
        log(f"  texture {asset} ({res})")
        for key, short in TEXTURE_MAPS.items():
            f = files[key][res]["jpg"]
            dest = root / "textures" / asset / f"{asset}_{short}_{res}.jpg"
            total += _download(f["url"], f["md5"], dest, log)
            entry["maps"][short] = str(dest.relative_to(root))
        index["textures"][asset] = entry
    for asset in want["models"]:
        files = _get_json(f"{API}/files/{asset}")
        b = files["blend"][MODEL_RESOLUTION]["blend"]
        log(f"  model {asset} ({MODEL_RESOLUTION})")
        dest = root / "models" / asset / f"{asset}_{MODEL_RESOLUTION}.blend"
        total += _download(b["url"], b["md5"], dest, log)
        for rel, f in b.get("include", {}).items():
            total += _download(f["url"], f["md5"], dest.parent / rel, log)
        index["models"][asset] = {"blend": str(dest.relative_to(root))}
    root.mkdir(parents=True, exist_ok=True)
    (root / "index.json").write_text(json.dumps(index, indent=1) + "\n", encoding="utf-8")
    index["downloaded_bytes"] = total
    return index


def resolved_assets(spec: BuildingSpec, root: Path) -> dict[str, Any] | None:
    """What the Blender stage needs: per material the texture maps and their
    real-world tile size, per plant the .blend file. None if nothing is
    fetched yet."""
    idx_path = root / "index.json"
    if not idx_path.exists():
        return None
    idx = json.loads(idx_path.read_text(encoding="utf-8"))
    from architectworks.units import to_metres

    mats = {}
    for m in spec.types.materials.values():
        t = m.texture
        if t is None or t.polyhaven not in idx["textures"]:
            continue
        e = idx["textures"][t.polyhaven]
        maps = {k: str((root / v).resolve()) for k, v in e["maps"].items() if (root / v).exists()}
        if "diff" not in maps:
            continue
        dims = e.get("dimensions_mm") or [1000, 1000]
        tile = to_metres(t.tile) if t.tile is not None else float(dims[0]) / 1000.0
        mats[m.unreal] = {"asset": t.polyhaven, "maps": maps, "tile_m": tile, "tint": t.tint, "value": t.value, "macro": t.macro}
    models = {k: str((root / v["blend"]).resolve()) for k, v in idx["models"].items() if (root / v["blend"]).exists()}
    return {"root": str(root.resolve()), "materials": mats, "models": models, "license": "CC0 (polyhaven.com)"}
