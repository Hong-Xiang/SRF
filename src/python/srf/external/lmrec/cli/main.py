import click
from srf.io.listmode import load_h5
from srf.external.lmrec.function import gen_script
import json

def lm2bin(source, target):
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
    lm2bin(source,target)
    gen_script(config,target,task)


@lmrec.command()
@click.option('--algorithm','-a',help='FBP2D or FBP3DRP or OSEM',type=click.Choice(['FBP2D','FBP3DRP','OSEM']))
@click.option('--config','-c',help='STIR config file',type=click.types.Path(False))
def execute(algorithm,config):
    pass
    

if __name__ == "__main__":
    lmrec()

