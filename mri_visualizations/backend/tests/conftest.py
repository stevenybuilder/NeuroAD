import pytest

from sfg import config
from sfg.fixtures import ensure_fixtures
from sfg.registry import Registry
from sfg.resources import ResourceStore


@pytest.fixture(scope="session")
def registry():
    ensure_fixtures()
    return Registry()


@pytest.fixture(scope="session")
def store(tmp_path_factory):
    return ResourceStore(tmp_path_factory.mktemp("resources"))


@pytest.fixture(scope="session")
def brats(registry):
    scans = registry.by_source("brats")
    if not scans:
        pytest.skip("no BraTS cases staged")
    return scans[0]


@pytest.fixture(scope="session")
def fixture_scans(registry):
    scans = {s.extra.get("defect"): s for s in registry.by_source("fixture")}
    if not scans:
        pytest.skip("no fixtures (needs IXI staged)")
    return scans


def _has_synthstrip():
    return config.SYNTHSTRIP_MODEL.exists()
