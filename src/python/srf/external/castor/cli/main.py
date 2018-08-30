import click
from srf.io.listmode import load_h5
from srf.external.castor.io import save_cdhf
from srf.external.castor.function import listmode2cdhf
# from srf.external.castor.function import listmode2cdhf, position2detectorid,generatesinogramspec,generatereconspec,get_scanner
# from srf.external.castor.data import (ReconstructionSpecScript, SinogramSpecScript)
from srf.external.castor.io import render
import json
from srf.data import PETCylindricalScanner
import os


def lm2castor(scanner, source, target):
    data = load_h5(source)
    cdh, cdf = listmode2cdhf(scanner, data)
    save_cdhf(target, cdh, cdf)

# def gen_sino_script(config,target):
#     sinogram_spec = generatesinogramspec(config,target)
#     sinogram_script = render(SinogramSpecScript(sinogram_spec))
#     save_script(target, sinogram_script)


def append_option(cmd: str, abbr_str, option_term):
    if option_term is None:
        cmd = cmd + ' '
    else:
        cmd = cmd +' ' + abbr_str + ' ' + option_term
    return cmd


@click.group()
def castor():
    """
    Interface to CASToR.
    """
    pass


@castor.command()
@click.option('--config', '-c', help='Config file', type=click.types.Path(True, dir_okay=False))
@click.option('--source', '-s', help='List mode data file', type=click.types.Path(True, dir_okay=False))
@click.option('--target', '-t', help='Target file path', type=click.types.Path(False))
def generate_data_and_header(config, source, target):
    pass
    # with open(config,'r') as fin:
    #     c = json.load(fin)
    # scanner = get_scanner(c['scanner']['petscanner'])
    # gen_sino_script(c['scanner']['petscanner'],target)
    # lm2sino(scanner,source,target)


@castor.command()
@click.option('--config', '-c', help='Config file', type=click.types.Path(True, dir_okay=False))
@click.option('--target', '-t', help='Target file path', type=click.types.Path(False))
@click.option('--source', '-s', help='Sinogram data file', type=click.types.Path(True, dir_okay=False), default=None)
def generate_recon_script(config, target, source):
    pass
    # with open(config,'r') as fin:
    #     c = json.load(fin)
    # if source is None:
    #     source = 'sinogram'
    # recon_spec = generatereconspec(c,source)
    # recon_script = render(ReconstructionSpecScript(recon_spec))
    # save_script(target,recon_script)


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
def recon(datafile, optimization, iteration, projector, convolution, dimension, voxel, dataoutput):
    execute = 'castor-recon '
    cmd = execute
    cmd = append_option(cmd, '-df', datafile)
    cmd = append_option(cmd, '-opti', optimization)
    cmd = append_option(cmd, '-it', iteration)
    cmd = append_option(cmd, '-proj', projector)
    cmd = append_option(cmd, '-conv', convolution)
    cmd = append_option(cmd, '-dim', dimension)
    cmd = append_option(cmd, '-vox', voxel)
    cmd = append_option(cmd, '-dout', dataoutput)
    print(cmd)
    os.system(cmd)


@castor.command()
@click.option('--inputfile', '-i', help='path to input root file', type=click.types.Path(False), default=None)
@click.option('--inputlist', '-il', help='path to input root file list', type=click.types.Path(False), default=None)
@click.option('--output', '-o', help='path to output files', type=click.types.Path(False), default='output_data')
@click.option('--macrofile', '-m', help='path to input macrofile', type=click.types.Path(False), default=None)
@click.option('--scanner', '-s', help='scanner alias in the configure folder')
def root2castor(inputfile, inputlist, output, macrofile, scanner):
    execute = 'castor-GATERootToCastor '
    cmd = execute
    if inputfile is None and inputlist is None:
        raise ValueError('No input root data file(s) for this convertion!')
    elif inputfile is not None and inputlist is not None:
        raise ValueError('Both input root data file and list are given!')
    elif inputfile is not None:
        cmd = cmd + ' -i ' + inputfile
    elif inputlist is not None:
        cmd = cmd + ' -il ' + inputlist
    else:
        pass

    cmd = append_option(cmd, '-o', output)
    cmd = append_option(cmd, '-m', macrofile)
    cmd = append_option(cmd, '-s', scanner)

    os.system(cmd)


@castor.command()
@click.option('--macrofile', '-m', help='path to input macrofile', type=click.types.Path(False), default=None)
@click.option('--scanner', '-o', help='scanner alias in the configure folder', type=str, default=None)
def mac2geom(macrofile, scanner):
    execute = 'castor-GATEMacToGeom '
    cmd = execute
    cmd = append_option(cmd, '-m', macrofile)
    cmd = append_option(cmd, '-o', scanner)
    print(cmd)
    os.system(cmd)


if __name__ == "__main__":
    castor()
