from ..data import (ScannerSpec,ReconstructionSpec,MapSpec,
    ReconstructionSpecScript,MapSpecScript,ScannerSpecScript)
import json
from srf.data import Block,PETCylindricalScanner
from jfs.api import Path


def gen_scanner_script(config,target):
    path_scanner_script = Path(target+'scanner_config.txt')
    scanner_spec = generatescannerspec(config)
    scanner_script = render(ScannerSpecScript(scanner_spec))
    save_script(path_scanner_script,scanner_script)

def gen_map_script(config,target):
    path_map_script = Path(target+'map_task.txt')
    map_spec = generatemapspec(config)
    map_script = render(MapSpecScript(map_spec))
    save_script(path_map_script,map_script)

def gen_recon_script(config,target):
    path_recon_script = Path(target+'recon_task.txt')
    recon_spec = generatereconspec(config,target)
    recon_script = render(ReconstructionSpecScript(recon_spec))
    save_script(path_recon_script,recon_script)

def gen_script(config,target,task):
    with open(config,'r') as fin:
        c = json.load(fin)
    gen_scanner_script(c)
    if task == 'recon':
        gen_recon_script(c,target)
    if task == 'map':
        gen_map_script(c,target)
    if task == 'both':
        gen_recon_script(c,target)
        gen_map_script(c,target)


def generatescannerspec(config):
    block = Block(config['scanner']['petscanner']['block']['size'],
                  config['scanner']['petscanner']['block']['grid'])
    return ScannerSpec(config['scanner']['petscanner']['ring']['inner_radius'],
                        config['scanner']['petscanner']['ring']['outer_radius'],
                        config['scanner']['petscanner']['ring']['axial_length'],
                        config['scanner']['petscanner']['ring']['nb_ring'],
                        config['scanner']['petscanner']['ring']['nb_block_per_ring'],
                        config['scanner']['petscanner']['ring']['gap'],
                        [block])


def generatereconspec(config):
    nb_subiterations,start_iteration = get_iter_info(config)
    tof_flag,tof_resolution,tof_binsize,tof_limit = get_tof_info(config)
    if ('abf' in config['algorithm']['correction']):
        abf_flag = 1
    else:
        abf_flag = 0 
    return ReconstructionSpec(config['output']['image']['grid'],
                              config['output']['image']['size'],
                              'input',
                              'output',
                              'map.ve',
                              start_iteration,
                              nb_subiterations,
                              tof_flag,
                              tof_resolution,
                              tof_binsize,
                              tof_limit,
                              abf_flag)


def generatemapspec(config):
    return MapSpec(config['output']['image']['grid'],
                   config['output']['image']['size'],
                   'map.ve')


def get_tof_info(config):
    if ('tof' in config['scanner']['petscanner']):
        tof_resolution = config['scanner']['petscanner']['tof']['resolution']
        tof_binsize = config['scanner']['petscanner']['tof']['bin']
        return 1,tof_resolution,tof_binsize,3
    else:
        return 0,0,0,0 

def get_iter_info(config):
    if ('osem' in config['algorithm']['recon']):
        nb_subiterations = config['algorithm']['recon']['osem']['nb_iterations']
        start_iteration = config['algorithm']['recon']['osem']['start_iteration']
    else:
        nb_subiterations = config['algorithm']['recon']['mlem']['nb_iterations']
        start_iteration = config['algorithm']['recon']['osem']['start_iteration']
    return nb_subiterations,start_iteration
