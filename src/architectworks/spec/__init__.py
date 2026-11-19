from pathlib import Path

import yaml

from architectworks.spec.model import BuildingSpec


def load_spec(path: str | Path) -> BuildingSpec:
    """Read and validate a Building Spec YAML file."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return BuildingSpec.model_validate(data)


__all__ = ["BuildingSpec", "load_spec"]
