from ..data import SinogramSpec,ReconstructionSpec,SinogramSpecScript
import json
from srf.data import Block,PETCylindricalScanner
from jfs.api import Path
from srf.external.stir.io import save_sinogram,save_script,render
from ._listmode2sinogram import listmode2sinogram



def lm2sino(scanner, data, target):
    sinogram = listmode2sinogram(scanner, data)
    save_sinogram(target, sinogram)

def gen_sino_script(config,target):
    sinogram_spec = generatesinogramspec(config,target)
    sinogram_script = render(SinogramSpecScript(sinogram_spec))
    save_script(target, sinogram_script)

def get_scanner(config):
    block = Block(config['block']['size'],
                  config['block']['grid'])
    return PETCylindricalScanner(config['ring']['inner_radius'],
                        config['ring']['outer_radius'],
                        config['ring']['axial_length'],
                        config['ring']['nb_rings'],
                        config['ring']['nb_blocks_per_ring'],
                        config['ring']['gap'],
                        [block])

def generatesinogramspec(config,path_sino):
    block = Block(config['block']['size'],
                  config['block']['grid'])
    path_sinogram = path_sino+'.s'
    return SinogramSpec(config['ring']['inner_radius'],
                        config['ring']['outer_radius'],
                        config['ring']['axial_length'],
                        config['ring']['nb_rings'],
                        config['ring']['nb_blocks_per_ring'],
                        config['ring']['gap'],
                        [block],
                        path_sinogram)

def generatereconspec(config,path_header):
    path_header = Path(path_header+'/sinogram.hs')
    part_config = config['algorithm']['recon']
    if ("osem" in part_config):
        nb_subsets = part_config['osem']['nb_subsets']
        nb_subiterations = part_config['osem']['nb_iterations']
    else:
        nb_subsets = 1
        nb_subiterations = part_config['mlem']['nb_iterations']
    
    return ReconstructionSpec(path_header.abs,
                              config['output']['image']['grid'][0],
                              nb_subsets,
                              nb_subiterations)

def get_algorithm(config):
    if ('fbp2d' in config):
        algorithm = 'FBP2D'
    elif ('fbp3drp' in config):
        algorithm = 'FBP3DRP'
    elif ('osem' in config and config['osem']['nb_subsets']!=1):
        algorithm = 'OSEM'
    else:
        algorithm = 'MLEM'
    return algorithm