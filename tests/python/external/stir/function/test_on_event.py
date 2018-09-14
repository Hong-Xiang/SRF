import pytest
import numpy as np
from srf.data import PositionEvent, DetectorIdEvent
from doufo import List
from dxl.shape.data import Point
from srf.external.stir.function import position2detectorid
from functools import partial


@pytest.fixture
def pos_and_ids(stir_data_root):
    data = np.load(stir_data_root / 'position2detectorids.npz')
    return {
        'input': List([PositionEvent(Point(r)) for r in data['position']]),
        'expect': List([DetectorIdEvent(r[0], r[1] // 10, r[1]) for r in data['ids']])
    }


def test_position2detectorid(scanner, pos_and_ids):
    result = pos_and_ids['input'].fmap(partial(position2detectorid, scanner))
    assert result == pos_and_ids['expect']
