import click
from srf.io.listmode import load_h5,save_h5
from srf.external.lmrec.function import gen_script
import json
import os

def lm2bin(source, target):
    """
    output bin file is float32, each event has 7 column, 6 cooridnate and a tof
    """
    data = load_h5(source)
    save_bin(target, bin_data)

@click.group()
def lmrec():
    """
    Interface to LMRec.
    """
    pass


@lmrec.command()
@click.option('--config', '-c', help='Config file', type=click.types.Path(True, dir_okay=False))
@click.option('--source', '-s', help='List mode data file', type=click.types.Path(True, dir_okay=False))
@click.option('--target', '-t', help='Target file path', type=click.types.Path(False))
@click.option('--task','-p',help='Recon or Map or both',type=click.Choice(['recon','map','both']))
def preprocess(config,source,target,task):          
    gen_script(config,target,task)
    #lm2bin(source,target)


@lmrec.command()
@click.option('--scanner','-s',help='Scanner config file',type=click.types.Path(False))
@click.option('--task','-t',help='Task file',type=click.types.Path(False))
def execute(algorithm,config):
    cmd = 'lm-recon '+scanner+' '+task
    os.system(cmd)

@lmrec.command()
@click.option('--source','-s',help='Reconstructed image files',type=click.types.Path(False))
@click.option('--target','-t',help='Output H5 file',type=click.types.Path(False))
def backprocess(source,target):
    pass
    

if __name__ == "__main__":
    lmrec()

