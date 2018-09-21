import click
from srf.external.stir.io import render,save_result,save_script
from srf.external.stir.function import generatereconspec,get_scanner,gen_sino_script,get_algorithm,lm2sino
from srf.external.stir.data import ReconstructionSpecScript
import json
from srf.io.listmode import load_h5
from srf.data import PETCylindricalScanner,LoR,PositionEvent
import os
import numpy as np
from jfs.api import Path
from doufo import List
from doufo.tensor import Vector
   

@click.group()
def stir():
    """
    Interface to STIR.
    """
    pass


@stir.command()
@click.option('--config', '-c', help='Config file', type=click.types.Path(True, dir_okay=False))
#@click.option('--source', '-s', help='List mode data file', type=click.types.Path(True, dir_okay=False))
@click.option('--target', '-t', help='Target file path', type=click.types.Path(False))
def generate_data_and_header(config,target):
    with open(config,'r') as fin:
        c = json.load(fin)
    scanner = get_scanner(c['scanner']['petscanner'])
    path_data = Path(target)
    gen_sino_script(c['scanner']['petscanner'],(path_data.abs+'sinogram'))
    path_file = c['input']['listmode']['path_file']
    data = load_h5(path_file)
    lor = List(LoR(PositionEvent(data['fst'][i,]),PositionEvent(data['snd'][i,]) ,weight=data['weight'][i]) for i in range(0,data['weight'].size))
    lm2sino(scanner,lor,(path_data.abs+'/sinogram.s'))


@stir.command()
@click.option('--config','-c',help='Config file',type = click.types.Path(True,dir_okay=False))
@click.option('--target','-t',help='Target file path',type=click.types.Path(False))
@click.option('--source','-s',help='Sinogram data file',type=click.types.Path(False),default=None)
def generate_recon_script(config,target,source):
    with open(config,'r') as fin:
        c = json.load(fin)
    if source is None:
        source = '.'
    recon_spec = generatereconspec(c,source)
    recon_script = render(ReconstructionSpecScript(recon_spec))
    path_script = Path(target).abs+'/OSMAPOSL.par'
    # if path_script.suffix == ' ':
    #     path_script = path_script+'.par'
    save_script(path_script,recon_script)

@stir.command()
@click.option('--config','-c',help='Config file',type = click.types.Path(True,dir_okay=False))
@click.option('--source','-s',help='STIR config file path',type=click.types.Path(False),default=None)
def execute(config,source):
    with open(config,'r') as fin:
        c = json.load(fin)
    part_config = c['algorithm']['recon']
    algorithm = get_algorithm(part_config)
    if algorithm=='MLEM' or 'OSEM':
        algorithm = 'OSMAPOSL'
    if source is not None:
        cmd = algorithm+' '+source+'/OSMAPOSL.par'
    else:
        cmd = algorithm
    print(cmd)
    os.system(cmd)

@stir.command()
@click.option('--config','-c',help='Config file',type=click.types.Path(True, dir_okay=False))
@click.option('--source','-s',help='STIR output file',type = click.types.Path(False))
def postprocess(config,source):
    with open(config,'r') as fin:
        c = json.load(fin)   
    save_result(c,source)

if __name__ == "__main__":
    stir()
