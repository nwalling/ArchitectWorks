from pathlib import Path

import pytest

from architectworks.resolve import resolve
from architectworks.spec import load_spec

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "projects" / "example-house" / "pavilion_a.yaml"


@pytest.fixture(scope="session")
def spec():
    return load_spec(EXAMPLE)


@pytest.fixture(scope="session")
def resolved(spec):
    return resolve(spec)
