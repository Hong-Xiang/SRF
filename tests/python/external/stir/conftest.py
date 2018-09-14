import pytest
import numpy as np
from srf.data import PETCylindricalScanner, Block


@pytest.fixture(scope='module')
def stir_data_root(test_data_root):
    return test_data_root / 'external' / 'stir'


@pytest.fixture(scope='module')
def l2sdata(stir_data_root):
    return np.load(stir_data_root / 'listmode2sinogram.npz')


@pytest.fixture(scope='module')
def scanner():
    # HACK FIXME rework to use config file
    return PETCylindricalScanner(
        inner_radius=99.0 / 2,
        outer_radius=119.0 / 2,
        axial_length=33.4,
        nb_rings=10,
        nb_blocks_per_ring=16,
        gap=0.0,
        blocks=[Block([20.0, 33.4, 3.34], [1, 10, 1])],
    )
