"""Lengths and angles as architects write them.

Lengths are kept as exact ``Fraction`` inches so that a spec value such as
``7'-7 3/4"`` compares character for character with the sheet and sums of
dimension strings close exactly. Conversion to metres happens once, at the
edge of the resolver.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

INCHES_PER_FOOT = 12
METRES_PER_INCH = Fraction(254, 10000)  # exact: 1" = 25.4 mm

_NUM = r"\d+(?:\.\d+)?"
_FRAC = r"\d+/\d+"
# feet part, then optional inches part: 8'-10", 7'-7 3/4", 26'-11 3/4", 5', 100'-0"
_FEET_INCH = re.compile(
    rf"""^\s*(?P<sign>-)?\s*
    (?P<feet>{_NUM})\s*'
    (?:\s*-?\s*
        (?P<inch>{_NUM})?
        (?:\s*[\s-]\s*(?P<frac>{_FRAC})|(?P<fraconly>{_FRAC}))?
        \s*"
    )?\s*$""",
    re.VERBOSE,
)
# inches only: 6", 5 1/2", 3/4", 1.5"
_INCH = re.compile(
    rf"""^\s*(?P<sign>-)?\s*
    (?:(?P<inch>{_NUM})(?:\s*[\s-]\s*(?P<frac>{_FRAC}))?|(?P<fraconly>{_FRAC}))
    \s*"\s*$""",
    re.VERBOSE,
)
_METRIC = re.compile(rf"^\s*(?P<sign>-)?\s*(?P<num>{_NUM})\s*(?P<unit>mm|cm|m)\s*$")
_METRIC_TO_INCH = {"mm": Fraction(1, 1) / Fraction(254, 10), "cm": Fraction(100, 254), "m": Fraction(10000, 254)}


class LengthError(ValueError):
    pass


def _frac(text: str | None) -> Fraction:
    if not text:
        return Fraction(0)
    return Fraction(text)


def parse_length(value: str | int | float | Fraction) -> Fraction:
    """Parse a written length into exact inches.

    Accepts feet-and-inches (``8'-10"``, ``7'-7 3/4"``, ``100'-0"``, ``5'``),
    inches (``6"``, ``5 1/2"``, ``3/4"``) and metric (``300 mm``, ``2.5 m``).
    Bare numbers are rejected: a length without a unit is a reading error.
    """
    if isinstance(value, Fraction):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        raise LengthError(f"length {value!r} has no unit; write it as on the drawing, e.g. 8'-10\" or 6\"")
    if not isinstance(value, str):
        raise LengthError(f"cannot read a length from {value!r}")
    text = value.replace("′", "'").replace("″", '"').replace("”", '"').replace("’", "'")
    m = _FEET_INCH.match(text)
    if m:
        feet = Fraction(m["feet"])
        inches = _frac(m["inch"]) + _frac(m["frac"]) + _frac(m["fraconly"])
        if inches >= INCHES_PER_FOOT:
            raise LengthError(f"{value!r}: inches part must be under 12")
        total = feet * INCHES_PER_FOOT + inches
        return -total if m["sign"] else total
    m = _INCH.match(text)
    if m:
        total = _frac(m["inch"]) + _frac(m["frac"]) + _frac(m["fraconly"])
        return -total if m["sign"] else total
    m = _METRIC.match(text)
    if m:
        total = Fraction(m["num"]) * _METRIC_TO_INCH[m["unit"]]
        return -total if m["sign"] else total
    raise LengthError(f"cannot read a length from {value!r}")


def format_length(inches: Fraction, denominator: int = 16) -> str:
    """Format exact inches the way a drafter would: ``7'-7 3/4"``.

    Values that are not a multiple of 1/denominator are rounded to the
    nearest 1/denominator inch.
    """
    sign = "-" if inches < 0 else ""
    inches = abs(Fraction(inches))
    rounded = Fraction(round(inches * denominator), denominator)
    feet, rem = divmod(rounded, INCHES_PER_FOOT)
    whole = int(rem)
    frac = rem - whole
    frac_text = f" {frac.numerator}/{frac.denominator}" if frac else ""
    if feet:
        return f"{sign}{int(feet)}'-{whole}{frac_text}\""
    if whole == 0 and frac:
        return f"{sign}{frac.numerator}/{frac.denominator}\""
    return f"{sign}{whole}{frac_text}\""


def to_metres(inches: Fraction) -> float:
    return float(Fraction(inches) * METRES_PER_INCH)


def to_feet(inches: Fraction) -> float:
    return float(Fraction(inches) / INCHES_PER_FOOT)


_ANGLE = re.compile(r"^\s*(?P<num>-?\d+(?:\.\d+)?)\s*(?:deg|°|degrees?)\s*$")


def parse_angle(value: str | int | float) -> float:
    """Parse ``30deg`` / ``30°`` into degrees (float)."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        raise LengthError(f"angle {value!r} has no unit; write e.g. 30deg")
    m = _ANGLE.match(str(value))
    if not m:
        raise LengthError(f"cannot read an angle from {value!r}")
    return float(m["num"])


def radians(deg: float) -> float:
    return math.radians(deg)
