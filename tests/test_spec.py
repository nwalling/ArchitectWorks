import copy

import pytest
import yaml
from pydantic import ValidationError

from architectworks.spec.model import BuildingSpec, json_schema
from tests.conftest import EXAMPLE


def raw():
    return yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))


def test_example_spec_loads(spec):
    assert spec.building.code == "EH"
    assert spec.levels["t_o_ring_bedroom"].elev == 110 * 12 + 9
    assert spec.elements[0].plan.side == 8 * 12 + 10


def test_every_value_has_a_source(spec):
    for name, lv in spec.levels.items():
        assert lv.src.basis
    for name, c in spec.types.components.items():
        assert c.src.basis, name
    assert spec.elements[0].plan.src.basis == "dimensioned"


def test_confidence_defaults_follow_basis(spec):
    assert spec.types.components["skylight_dome"].src.confidence == "low"  # measured
    assert spec.levels["ff_bedroom"].src.confidence == "high"  # dimensioned


def test_unknown_reference_is_reported():
    data = raw()
    data["elements"][0]["header"]["top"] = "no_such_level"
    with pytest.raises(ValidationError, match="unknown level 'no_such_level'"):
        BuildingSpec.model_validate(data)


def test_wrong_type_kind_is_reported():
    data = raw()
    data["elements"][0]["columns"]["type"] = "header_6x6"
    with pytest.raises(ValidationError, match="is a beam, expected a column"):
        BuildingSpec.model_validate(data)


def test_unitless_length_is_rejected():
    data = raw()
    data["elements"][0]["plan"]["side"] = 106
    with pytest.raises(ValidationError, match="has no unit"):
        BuildingSpec.model_validate(data)


def test_unknown_field_is_rejected():
    data = raw()
    data["elements"][0]["colour"] = "red"
    with pytest.raises(ValidationError):
        BuildingSpec.model_validate(data)


def test_glazed_sides_must_be_contiguous():
    data = copy.deepcopy(raw())
    data["elements"][0]["glazed_sides"] = [0, 2]
    with pytest.raises(ValidationError, match="contiguous"):
        BuildingSpec.model_validate(data)


def test_json_schema_is_generated():
    schema = json_schema()
    assert "BuildingSpec" in schema.get("title", "") or "properties" in schema
    assert "elements" in schema["properties"]
