import pytest
import numpy as np
from srf.data import ListModeData, PETSinogram3D, PETCylindricalScanner, Block
from srf.external.stir.function import ndarray2listmode, listmode2sinogram, position2detectorid
from srf.external.stir.function._listmode2sinogram import rework_indices, id_sinogram, id_view, id_bin
from functools import partial
from dxl.function import x
from dxl.data import List


@pytest.fixture
def listmode_data(scanner, l2sdata):
    return ndarray2listmode(l2sdata['input']).fmap(lambda p: p.fmap2(partial(position2detectorid, scanner)))


@pytest.fixture
def l2s_expected_result(l2sdata):
    return PETSinogram3D(l2sdata['result'])


def test_listmode_to_sinogram(scanner, listmode_data, l2s_expected_result):
    sinogram = listmode2sinogram(scanner, listmode_data)
    assert sinogram.shape == l2s_expected_result.shape
    assert sinogram == l2s_expected_result


@pytest.fixture
def ring_ids(scanner, listmode_data):
    lors = listmode_data.fmap(partial(rework_indices, scanner))
    return lors.fmap(lambda l: l.fmap2(lambda e: e.id_ring))


@pytest.fixture
def crystal_ids(scanner, listmode_data):
    lors = listmode_data.fmap(partial(rework_indices, scanner))
    return lors.fmap(lambda l: l.fmap2(lambda e: e.id_crystal))


@pytest.fixture
def id_sinogram_matlab(stir_data_root):
    data = np.load(stir_data_root / 'sinogramids.npz')
    return {k: List((v - 1).tolist()) for k, v in data.items()}


def test_id_sinogram(scanner, ring_ids, stir_data_root, id_sinogram_matlab):
    result = ring_ids.fmap(partial(id_sinogram, scanner))
    assert result == id_sinogram_matlab['id_sinogram']


def test_id_bin(scanner, crystal_ids, stir_data_root, id_sinogram_matlab):
    result = crystal_ids.fmap(partial(id_bin, scanner))
    assert result == id_sinogram_matlab['id_bin']


def test_id_view(scanner, crystal_ids, stir_data_root, id_sinogram_matlab):
    result = crystal_ids.fmap(partial(id_view, scanner))
    assert result == id_sinogram_matlab['id_view']
