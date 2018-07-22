import pytest
import numpy as np
from srf.data import ListModeData, PETSinogram3D, PETCylindricalScanner, Block
from srf.external.stir.function import ndarray2listmode, listmode2sinogram


@pytest.fixture
def listmode_data(l2sdata):
    return ndarray2listmode(l2sdata['input'])


@pytest.fixture
def l2s_expected_result(l2sdata):
    return PETSinogram3D(l2sdata['result'])


@pytest.mark.skip('NIY')
def test_listmode_to_sinogram(scanner, listmode_data):
    sinogram = listmode2sinogram(scanner, listmode_data)
