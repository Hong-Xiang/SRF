import click
import os
import json
from srf.io.listmode import load_h5
# from srf.external.castor.io import save_cdhf
from srf.external.castor.function import listmode2cdhf
from srf.external.castor.function import make_mac
# from srf.external.castor.function import listmode2cdhf, position2detectorid,generatesinogramspec,generatereconspec,get_scanner
# from srf.external.castor.data import (ReconstructionSpecScript, SinogramSpecScript)
# from srf.external.castor.io import render
# from srf.data import PETCylindricalScanner
from .parser import parse_recon, parse_root_to_castor, parse_mac_to_geom, parse_listmode_to_castor, parse_sino_to_castor





# def gen_sino_script(config,target):
#     sinogram_spec = generatesinogramspec(config,target)
#     sinogram_script = render(SinogramSpecScript(sinogram_spec))
#     save_script(target, sinogram_script)



@click.group()
def castor():
    """
    Interface to CASToR.
    """
    pass


@castor.command()
@click.option('--config', '-c', help= 'path to the configuration file', type=click.types.Path(False))
def recon(config):

    cmd = parse_recon(config)
    os.system(cmd)


@castor.command()
@click.option('--config', '-c', help= 'path to the configuration file', type=click.types.Path(False))
def root2castor(config):
    cmd = parse_root_to_castor(config)
    os.system(cmd)


@castor.command()
@click.option('--config', '-c', help= 'path to the configuration file', type=click.types.Path(False))
def mac2geom(config):
    cmd = parse_mac_to_geom(config)
    os.system(cmd)

@castor.command()
@click.option('--config', '-c', help = 'path to the GATE sinogram to CASToR list-mode data', type=click.types.Path(False))
def sino2cdhf(config):
    cmd = parse_sino_to_castor(config)
    os.system(cmd)

@castor.command()
@click.option('--config', '-c', help= 'path to the configuration file', type=click.types.Path(False))
def lm2cdhf(config):
    config = parse_listmode_to_castor(config)
    listmode2cdhf(config)

@castor.command()
@click.option('--config', '-c', help='path to the configuration file', type=click.types.Path(False))
def geo2mac(config:dict):
    from srf.external.castor.io import load_config
    make_mac(load_config(config))


if __name__ == "__main__":
    castor()
