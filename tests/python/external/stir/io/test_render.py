import pytest
from srf.external.stir.data import (
    ReconstructionSpec, ReconstructionSpecScript, SinogramSpecScript, SinogramSpec)
from srf.external.stir.io import render
from srf.data import Block


@pytest.fixture
def recon_spec():
    return ReconstructionSpec('brain.hs', 40, 10, 50)


@pytest.fixture
def sinogram_spec():
    return SinogramSpec(99.39, 119.0 / 2, 33.4, 10, 8, 0.0, Block([20.0, 33.4, 3.34], [1, 10, 10]), 'brain.s')


@pytest.fixture
def expect_recon_script(stir_data_root):
    with open(stir_data_root / 'OSMapOSL.par', 'r') as fin:
        return fin.read()


@pytest.fixture
def expect_sinogram_script(stir_data_root):
    with open(stir_data_root / 'sinogram.hs', 'r') as fin:
        return fin.read()


def test_render_reconstruction_spec(recon_spec, expect_recon_script):
    result = render(ReconstructionSpecScript(recon_spec))
    assert result.strip() == expect_recon_script.strip()


def test_render_sinogram_spec(sinogram_spec, expect_sinogram_script):
    result = render(SinogramSpecScript(sinogram_spec))
    assert result.strip() == expect_sinogram_script.strip()
