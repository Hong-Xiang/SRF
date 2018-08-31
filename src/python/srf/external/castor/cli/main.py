import click
import os
import json
from srf.io.listmode import load_h5
from srf.external.castor.io import save_cdhf
from srf.external.castor.function import listmode2cdhf
# from srf.external.castor.function import listmode2cdhf, position2detectorid,generatesinogramspec,generatereconspec,get_scanner
# from srf.external.castor.data import (ReconstructionSpecScript, SinogramSpecScript)
from srf.external.castor.io import render
from srf.data import PETCylindricalScanner
from .parser import parse_recon, parse_root_to_castor, parse_mac_to_geom


def lm2castor(scanner, source, target):
    data = load_h5(source)
    cdh, cdf = listmode2cdhf(scanner, data)
    save_cdhf(target, cdh, cdf)

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
@click.option('--datafile', '-df', help='path to cdh data file', type=click.types.Path(False), default=None)
@click.option('--optimization', '-opti', help='optimization method type',
              type=click.Choice(['AML', 'LDWB', 'MLEM', 'MLTR', 'NEGML']), default=None)
@click.option('--iteration', '-it', help='number of iterations: nb_iters:nb_subsets', type=str)
@click.option('--projector', '-proj', help='projector type',
              type=click.Choice(['classicSiddon', 'distanceDriven', 'incrementalSiddon', 'joseph', 'multiSiddon']), default=None)
@click.option('--convolution', '-conv', help='convolution setting: filter_type,ix,iy,iz::psf', type=str, default=None)
@click.option('--dimension', '-dim', help='reconstruction image dimension: nx,ny,nz', type=str, default=None)
@click.option('--voxel', '-vox', help='voxel size: vx,vy,vz', type=str, default=None)
@click.option('--dataoutput', '-dout', help='output image file name', type=str, default=None)

@castor.command()
@click.option('--config', '-c', help= 'path to the configuration file', type=click.type.Path(False))
def recon(config):


    # print(cmd)
    os.system(cmd)


@castor.command()
@click.option('--inputfile', '-i', help='path to input root file', type=click.types.Path(False), default=None)
@click.option('--inputlist', '-il', help='path to input root file list', type=click.types.Path(False), default=None)
@click.option('--output', '-o', help='path to output files', type=click.types.Path(False), default='output_data')
@click.option('--macrofile', '-m', help='path to input macrofile', type=click.types.Path(False), default=None)
@click.option('--scanner', '-s', help='scanner alias in the configure folder')



def root2castor(inputfile, inputlist, output, macrofile, scanner):


    os.system(cmd)


@castor.command()
@click.option('--macrofile', '-m', help='path to input macrofile', type=click.types.Path(False), default=None)
@click.option('--scanner', '-o', help='scanner alias in the configure folder', type=str, default=None)
def mac2geom(macrofile, scanner):

    os.system(cmd)


if __name__ == "__main__":
    castor()
