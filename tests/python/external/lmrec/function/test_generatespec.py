from srf.external.lmrec.function.genespec import (gen_scanner_script,gen_map_script,
       gen_recon_script,gen_script,generatescannerspec,generatereconspec,
       generatemapspec,get_tof_info,get_iter_info)
from srf.external.lmrec.data import (ScannerSpec,ReconstructionSpec,MapSpec,
    ReconstructionSpecScript,MapSpecScript,ScannerSpecScript)
from srf.data import Block
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

def test_get_tof_info(get_config):
    tof_flag,tof_resolution,tof_binsize,tof_limit = get_tof_info(get_config)
    assert tof_flag == 1
    assert tof_resolution == 200
    assert tof_binsize == 3
    assert tof_limit == 3

def test_get_iter_info(get_config):
    nb_subiterations,start_iteration = get_iter_info(get_config)
    assert nb_subiterations == 10
    assert start_iteration == 3

def test_generate_map_spec(get_config):
    result = generatemapspec(get_config)
    assert result == MapSpec([64,64,64],[128.0,128.0,128.0],'map.ve')

def test_generate_recon_spec(get_config):
    result = generatereconspec(get_config)
    assert result == ReconstructionSpec([64,64,64],[128.0,128.0,128.0],'input',
         'output','map.ve',3,10,1,200,3,3,0)

def test_generate_scanner_spec(get_config):
    result = generatescannerspec(get_config)
    assert result == ScannerSpec(30.0,40.0,25.5,10,30,0.3,[Block([64.0,64.0,64.0],[30,30,30])])
