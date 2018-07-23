import pytest
from srf.external.stir.data import ReconstructionSpec, ReconstructionSpecScript
from srf.external.stir.io import render


@pytest.fixture
def recon_spec():
    return ReconstructionSpec('brain.hs', 40, 10, 50)

@pytest.fixture
def expect_recon_script(stir_data_root):
    with open(stir_data_root/'OSMapOSL.par', 'r') as fin:
        return fin.read()

def test_render_reconstruction(recon_spec, expect_recon_script):
    result = render(ReconstructionSpecScript(recon_spec))
    assert result.strip() == expect_recon_script.strip()
