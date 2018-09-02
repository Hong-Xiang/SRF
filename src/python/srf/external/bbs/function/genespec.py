from srf.external.lmrec.data import (MapSpecScript,ScannerSpecScript,ReconstructionSpec,
                                TOFSpec,IterSpec,ReconstructionSpecScript)
from srf.external.lmrec.function import (generatescannerspec,generatereconspec,generatemapspec,
                                  gen_scanner_script)
from jfs.api import Path
from ..io import render,save_script


def gen_map_script(config,target):
    path_map_script = Path(target+'map_task.txt')
    map_spec = generatemapspec(config['output']['image'])
    map_script = render(MapSpecScript(map_spec))
    save_script(path_map_script,map_script)

def gen_recon_script(config,target):
    path_recon_script = Path(target+'recon_task.txt')
    recon_spec = generatereconspec(config)
    recon_script = render(ReconstructionSpecScript(recon_spec))
    save_script(path_recon_script,recon_script)

def gen_script(config,target,task):
    gen_scanner_script(config,target)
    if task == 'recon':
        gen_recon_script(config,target)
    if task == 'map':
        gen_map_script(config,target)
    if task == 'both':
        gen_recon_script(config,target)
        gen_map_script(config,target)

def generatereconspec(config):
    iter_spec = get_iter_info(config['algorithm']['recon'])
    tof_spec = get_tof_info(config['scanner']['petscanner'])
    if ('abf' in config['algorithm']['correction']):
        abf_flag = 1
    else:
        abf_flag = 0 
    return ReconstructionSpec(config['output']['image']['grid'],
                              config['output']['image']['size'],
                              'input.txt',
                              'output',
                              'map.ve',
                              iter_spec.start_iteration,
                              iter_spec.nb_subiterations,
                              tof_spec.tof_flag,
                              tof_spec.tof_resolution,
                              tof_spec.tof_binsize,
                              tof_spec.tof_limit,
                              abf_flag)

def get_tof_info(config):
    if ('tof' in config):
        spec = TOFSpec(1,config['tof']['resolution'],config['tof']['bin'],3) 
    else:
        spec = TOFSpec(0,0,0,0)
    return spec

def get_iter_info(config):
    if ('osem' in config):
        spec = IterSpec(config['osem']['nb_iterations'],config['osem']['start_iteration'])
    else:
        spec = IterSpec(config['mlem']['nb_iterations'],config['mlem']['start_iteration'])
    return spec