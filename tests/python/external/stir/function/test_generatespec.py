from srf.external.stir.function import get_scanner,generatesinogramspec,generatereconspec
from srf.data import PETCylindricalScanner,Block
from srf.external.stir.data import SinogramSpec,ReconstructionSpec
import pytest

@pytest.fixture
def get_config():
    return  {
    "scanner": {
        "petscanner": {
            "ring": {
                "inner_radius": 30.0,
                "outer_radius": 40.0,
                "axial_length": 25.5,
                "nb_ring": 10,
                "nb_block_per_ring": 30,
                "gap": 0.3
            },
            "block": {
                "grid": [
                    30,
                    30,
                    30
                ],
                "size": [
                    64.0,
                    64.0,
                    64.0
                ],
                "interval": [
                    0.2,
                    0.2,
                    0.2
                ]
            },
            "tof": {
                "resolution": 200,
                "bin": 3
            }
        }
    },
    "input": {
        "sinogram1": {
            "path_file": "/home/twj2417/SRF/Input.npz",
            "path_dataset": "sinogram1",
            "slice": "[1:10,:]"
        },
        "system_matrix": {
            "path_file": "/home/twj2417/SRF/Input.npz",
            "path_dataset": "sm"
        }
    },
    "output": {
        "image": {
            "grid": [
                64,
                64,
                64
            ],
            "size": [
                128.0,
                128.0,
                128.0
            ],
            "center": [
                0.0,
                0.0,
                0.0
            ],
            "map_file": {
                "path_file": "/home/twj2417/SRF/Map.npy"
            },
        "path_file":"./result.npz",
        "path_dataset_prefix": "result"
        }
        
    },
    "algorithm": {
        "projection_model": {
            "tor": {
                "kernel_width": 4.0
            }
        },
        "recon": {
            "osem": {
                "nb_subsets": 5,
                "nb_iterations": 10,
                "start_iteration": 3,
                "save_interval": 1
            }
        },
        "correction": {
            "atten_correction": {
                "map_file": "/home/twj2417/SRF/Atten.npy"
            }
        }
    },
    "__version__":"0.0.1"
}

@pytest.fixture
def get_target():
    return '/tmp/input'

def test_get_scanner(get_config):
    result = get_scanner(get_config)
    assert result == PETCylindricalScanner(30.0,40.0,25.5,10,30,0.3,[Block([64.0,64.0,64.0],[30,30,30])])

def test_gen_sino_spec(get_config,get_target):
    result = generatesinogramspec(get_config,get_target)
    assert result == SinogramSpec(30.0,40.0,25.5,10,30,0.3,[Block([64.0,64.0,64.0],[30,30,30])],'/tmp/input.s')
     
def test_gen_recon_spec(get_config,get_target):
    result = generatereconspec(get_config,get_target)
    assert result == ReconstructionSpec('/tmp/input.hs',64,5,10)