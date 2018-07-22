import pytest
import numpy as np


@pytest.fixture(scope='module')
def l2sdata(test_data_root):
    return np.load(test_data_root / 'external' /
                   'stir' / 'listmode2sinogram.npz')
