import pytest
import os
from pathlib import Path


def pytest_collection_modifyitems(session, config, items):
    items[:] = [item for item in items if item.name != 'test_session']


@pytest.fixture(scope='module')
def test_data_root():
    return Path(os.environ.get('GROOT', '/mnt/gluster')) / 'CustomerTests' / 'SRF'
