import click
from srf.io.listmode import load_h5 
from srf.external.stir.io import save_sinogram,save_script
from srf.external.stir.function import listmode2sinogram, position2detectorid,generatesinogramspec,generatereconspec,get_scanner
from srf.external.stir.data import (ReconstructionSpecScript, SinogramSpecScript)
from srf.external.stir.io import render
import json
from srf.data import PETCylindricalScanner
import os


def lm2sino(scanner, source, target):
    data = load_h5(source)
    sinogram = listmode2sinogram(scanner, data)
    save_sinogram(target, sinogram)

def gen_sino_script(config,target):
    sinogram_spec = generatesinogramspec(config,target)
    sinogram_script = render(SinogramSpecScript(sinogram_spec))
    save_script(target, sinogram_script)
    

@click.group()
def stir():
    """
    Interface to STIR.
    """
    pass


@stir.command()
@click.option('--config', '-c', help='Config file', type=click.types.Path(True, dir_okay=False))
@click.option('--source', '-s', help='List mode data file', type=click.types.Path(True, dir_okay=False))
@click.option('--target', '-t', help='Target file path', type=click.types.Path(False))
def generate_data_and_header(config,source,target):
    with open(config,'r') as fin:
        c = json.load(fin)
    scanner = get_scanner(c['scanner']['petscanner'])
    gen_sino_script(c['scanner']['petscanner'],target)
    lm2sino(scanner,source,target)


@stir.command()
@click.option('--config','-c',help='Config file',type = click.types.Path(True,dir_okay=False))
@click.option('--target','-t',help='Target file path',type=click.types.Path(False))
@click.option('--source','-s',help='Sinogram data file',type=click.types.Path(True,dir_okay=False),default=None)
def generate_recon_script(config,target,source):
    with open(config,'r') as fin:
        c = json.load(fin)
    if source is None:
        source = 'sinogram'
    recon_spec = generatereconspec(c,source)
    recon_script = render(ReconstructionSpecScript(recon_spec))
    save_script(target,recon_script)

@stir.command()
@click.option('--algorithm','-a',help='FBP2D or FBP3DRP or OSEM',type=click.Choice(['FBP2D','FBP3DRP','OSEM']))
@click.option('--config','-c',help='STIR config file',type=click.types.Path(False),default=None)
def execute(algorithm,config):
    if config is not None:
        cmd = algorithm+' '+config
    else:
        cmd = algorithm
    os.system(cmd)


if __name__ == "__main__":
    stir()
