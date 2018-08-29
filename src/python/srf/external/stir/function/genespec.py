from ..data import SinogramSpec,ReconstructionSpec
import json
from srf.data import Block,PETCylindricalScanner
from jfs.api import Path


def get_scanner(config):
    block = Block(config['block']['size'],
                  config['block']['grid'])
    return PETCylindricalScanner(config['ring']['inner_radius'],
                        config['ring']['outer_radius'],
                        config['ring']['axial_length'],
                        config['ring']['nb_ring'],
                        config['ring']['nb_block_per_ring'],
                        config['ring']['gap'],
                        [block])

def generatesinogramspec(config,path_sino):
    block = Block(config['block']['size'],
                  config['block']['grid'])
    path_sinogram = Path(path_sino+'.s')
    return SinogramSpec(config['ring']['inner_radius'],
                        config['ring']['outer_radius'],
                        config['ring']['axial_length'],
                        config['ring']['nb_ring'],
                        config['ring']['nb_block_per_ring'],
                        config['ring']['gap'],
                        [block],
                        path_sinogram.abs)

def generatereconspec(config,path_header):
    path_header = Path(path_header+'.hs')
    if ("osem" in config['algorithm']['recon']):
        nb_subsets = config['algorithm']['recon']['osem']['nb_subsets']
        nb_subiterations = config['algorithm']['recon']['osem']['nb_iterations']
    else:
        nb_subsets = 1
        nb_subiterations = config['algorithm']['recon']['mlem']['nb_iterations']
    
    return ReconstructionSpec(path_header.abs,
                              config['output']['image']['grid'][0],
                              nb_subsets,
                              nb_subiterations)
