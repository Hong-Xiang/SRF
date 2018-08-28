from ..data import SinogramSpec,ReconstructionSpec
import json
from srf.data import Block,PETCylindricalScanner
from jfs.api import Path


def get_scanner(config):
    block = Block(config['scanner']['petscanner']['block']['size'],
                  config['scanner']['petscanner']['block']['grid'])
    return PETCylindricalScanner(config['scanner']['petscanner']['ring']['inner_radius'],
                        config['scanner']['petscanner']['ring']['outer_radius'],
                        config['scanner']['petscanner']['ring']['axial_length'],
                        config['scanner']['petscanner']['ring']['nb_ring'],
                        config['scanner']['petscanner']['ring']['nb_block_per_ring'],
                        config['scanner']['petscanner']['ring']['gap'],
                        [block])

def generatesinogramspec(config,path_sino):
    block = Block(config['scanner']['petscanner']['block']['size'],
                  config['scanner']['petscanner']['block']['grid'])
    path_sinogram = Path(path_sino+'.s')
    return SinogramSpec(config['scanner']['petscanner']['ring']['inner_radius'],
                        config['scanner']['petscanner']['ring']['outer_radius'],
                        config['scanner']['petscanner']['ring']['axial_length'],
                        config['scanner']['petscanner']['ring']['nb_ring'],
                        config['scanner']['petscanner']['ring']['nb_block_per_ring'],
                        config['scanner']['petscanner']['ring']['gap'],
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
