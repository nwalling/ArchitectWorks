from fractions import Fraction

import pytest

from architectworks.units import LengthError, format_length, parse_angle, parse_length, to_metres


@pytest.mark.parametrize(
    "text, inches",
    [
        ("8'-10\"", 106),
        ("7'-7 3/4\"", Fraction(367, 4)),
        ("26'-11 3/4\"", Fraction(1295, 4)),
        ("100'-0\"", 1200),
        ("5'", 60),
        ("6\"", 6),
        ("5 1/2\"", Fraction(11, 2)),
        ("3/4\"", Fraction(3, 4)),
        ("6'-7 1/2\"", Fraction(159, 2)),
        ("107'-1 1/2\"", Fraction(2571, 2)),
        ("-2'-0\"", -24),
        ("7'-7-3/4\"", Fraction(367, 4)),
        ("1.5\"", Fraction(3, 2)),
        ("300 mm", Fraction(300 * 10, 254)),
        ("2.54 cm", 1),
    ],
)
def test_parse_length(text, inches):
    assert parse_length(text) == inches


def test_written_forms_parse_exactly():
    assert parse_length("7'-7 3/4\"") == 7 * 12 + 7 + Fraction(3, 4)
    assert parse_length("26'-11 3/4\"") == 26 * 12 + 11 + Fraction(3, 4)


@pytest.mark.parametrize("bad", ["8-10", "10", 10, 2.5, "8'-12\"", "eight feet", ""])
def test_rejects_unitless_and_malformed(bad):
    with pytest.raises(LengthError):
        parse_length(bad)


@pytest.mark.parametrize("text", ["8'-10\"", "7'-7 3/4\"", "6\"", "3/4\"", "5 1/2\"", "100'-0\"", "-2'-3\""])
def test_format_round_trips(text):
    assert format_length(parse_length(text)) == text


def test_foundation_plan_chains_close_exactly():
    # Read off the foundation plan photo: parts must sum to the overall above them.
    p = parse_length
    assert p("5'-8\"") + p("8'-10\"") + p("4'-10\"") + p("7'-7 3/4\"") == p("26'-11 3/4\"")
    assert p("8'-10\"") + p("4'-10\"") + p("7'-7 3/4\"") == p("21'-3 3/4\"")
    assert p("7'-7 3/4\"") + p("5'-11\"") == p("13'-6 3/4\"")
    assert p("5'-11\"") + p("2'-4\"") + p("3'-5\"") == p("11'-8\"")
    assert p("13'-6 3/4\"") + p("11'-8\"") == p("25'-2 3/4\"")


def test_metres_are_exact_at_the_inch():
    assert to_metres(parse_length("1\"")) == pytest.approx(0.0254, abs=1e-15)
    assert to_metres(parse_length("8'-10\"")) == pytest.approx(2.6924, abs=1e-12)


def test_angles():
    assert parse_angle("30deg") == 30.0
    assert parse_angle("7.5deg") == 7.5
    assert parse_angle("30°") == 30.0
    with pytest.raises(LengthError):
        parse_angle(30)
