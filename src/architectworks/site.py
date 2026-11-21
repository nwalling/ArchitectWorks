"""The site: where the sun is, which way is north, and what the resolver
hands Blender (and later Unreal) to build the ground and planting from."""

from __future__ import annotations

import datetime as dt
import math
from typing import Any

from shapely.ops import unary_union

from architectworks.spec.model import BuildingSpec
from architectworks.units import to_metres


def sun_position(latitude: float, longitude: float, utc_offset: float, date: str, time: str) -> tuple[float, float]:
    """(azimuth, elevation) in degrees; azimuth clockwise from true north.
    NOAA's simplified solar position, good to a fraction of a degree."""
    d = dt.date.fromisoformat(date)
    hh, mm = (int(x) for x in time.split(":"))
    day = d.timetuple().tm_yday
    hour = hh + mm / 60.0
    g = 2 * math.pi / (366 if d.year % 4 == 0 and (d.year % 100 or d.year % 400 == 0) else 365) * (day - 1 + (hour - 12) / 24)
    eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g) - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))
    decl = 0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g) + 0.000907 * math.sin(2 * g) - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g)
    time_offset = eqtime + 4 * longitude - 60 * utc_offset
    tst = hh * 60 + mm + time_offset
    ha = math.radians(tst / 4 - 180)
    lat = math.radians(latitude)
    cos_zen = math.sin(lat) * math.sin(decl) + math.cos(lat) * math.cos(decl) * math.cos(ha)
    zen = math.acos(max(-1.0, min(1.0, cos_zen)))
    el = 90 - math.degrees(zen)
    az = math.degrees(math.atan2(math.sin(ha), math.cos(ha) * math.sin(lat) - math.tan(decl) * math.cos(lat))) + 180
    return az % 360, el


def plan_direction(true_north_deg: float, azimuth_deg: float) -> tuple[float, float]:
    """Plan unit vector for a compass bearing (clockwise from true north)."""
    a = math.radians(true_north_deg - azimuth_deg)
    return math.cos(a), math.sin(a)


def resolve_site(spec: BuildingSpec, footprints: list) -> dict[str, Any] | None:
    site = spec.site
    if site is None:
        return None
    t = site.terrain
    datum = spec.levels[spec.datum].elev
    az, el = sun_position(site.location.latitude, site.location.longitude, site.location.utc_offset, site.sun.date, site.sun.time)
    sx, sy = plan_direction(site.true_north, az)
    outline = unary_union([fp.polygon for fp in footprints]).buffer(0.3).simplify(0.05)
    # exteriors and holes alike: an enclosed courtyard is outside the building
    rings = [list(map(list, r.coords)) for g in getattr(outline, "geoms", [outline]) for r in (g.exterior, *g.interiors)]
    return {
        "location": site.location.model_dump(),
        "true_north_deg": float(site.true_north),
        "north": list(plan_direction(site.true_north, 0.0)),
        "grade_m": to_metres(spec.levels[site.grade].elev - datum),
        "sun": {"date": site.sun.date, "time": site.sun.time, "azimuth_deg": round(az, 2), "elevation_deg": round(el, 2), "direction": [sx * math.cos(math.radians(el)), sy * math.cos(math.radians(el)), math.sin(math.radians(el))]},
        "terrain": {
            "pad_m": to_metres(t.pad),
            "blend_m": to_metres(t.blend),
            "extent_m": to_metres(t.extent),
            "relief_m": to_metres(t.relief),
            "wavelength_m": to_metres(t.wavelength),
            "fall": [math.cos(math.radians(t.fall)), math.sin(math.radians(t.fall))],
            "grade": t.grade,
            "hills_m": to_metres(t.hills),
            "seed": t.seed,
            "ground": spec.types.materials[t.ground].unreal,
        },
        "planting": [
            {"polyhaven": p.polyhaven, "count": p.count, "near_m": to_metres(p.near), "far_m": to_metres(p.far), "scale": list(p.scale), "spacing_m": to_metres(p.spacing), "controls": dict(p.controls)}
            for p in site.planting
        ],
        "outline": rings,
        "source": site.src.model_dump(exclude_none=True),
    }
