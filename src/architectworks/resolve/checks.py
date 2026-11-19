"""Cross-checks the drawings already contain: chains that must close, typical
dimensions that must follow their rule."""

from __future__ import annotations

import math
from fractions import Fraction

from architectworks.resolve.model import Check
from architectworks.spec.model import BuildingSpec
from architectworks.units import format_length


def run_spec_checks(spec: BuildingSpec) -> list[Check]:
    out: list[Check] = []
    for i, ch in enumerate(spec.checks.chains):
        total = sum(ch.parts, Fraction(0))
        diff = total - ch.overall
        ok = abs(diff) <= ch.tolerance
        parts = " + ".join(format_length(p) for p in ch.parts)
        out.append(
            Check(
                id=f"chain.{ch.sheet}.{i}",
                kind="chain",
                status="pass" if ok else "conflict",
                message=f"{ch.sheet} ({ch.where}): {parts} = {format_length(total)}, written {format_length(ch.overall)}",
                values={"sum": format_length(total), "overall": format_length(ch.overall), "difference": format_length(diff)},
            )
        )
    for i, r in enumerate(spec.checks.rules):
        side_in = float(r.side)
        if r.rule == "hexagon_apothem":
            expected = side_in * math.cos(math.pi / 6)
            label = f"apothem of a {format_length(r.side)} hexagon"
        else:
            expected = side_in / 2
            label = f"half of a {format_length(r.side)} side"
        diff = float(r.dimensioned) - expected
        ok = abs(diff) <= float(r.tolerance)
        out.append(
            Check(
                id=f"rule.{r.rule}.{i}",
                kind="rule",
                status="pass" if ok else "conflict",
                message=f"{r.sheet}: {label} is {expected / 12:.4f} ft; written {format_length(r.dimensioned)} (off by {diff:+.3f} in)",
                values={"expected_in": round(expected, 4), "written": format_length(r.dimensioned), "difference_in": round(diff, 4)},
            )
        )
    return out
