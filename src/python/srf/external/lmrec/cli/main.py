import click
from jfs.api import Path
from srf.io.listmode import load_h5
from srf.external.lmrec.function import gen_script
from srf.external.lmrec.io import save_result
import json
import os
import numpy as np

def lm2bin(config, target):
    path_file = config['input']['listmode']['path_file']
    data = load_h5(path_file)
    coo_data = np.hstack((data['fst'],data['snd']))
    bin_data = np.hstack((coo_data,data['weight'].reshape(data['weight'].size,1)))
    out = bin_data.astype(np.float32)
    out.tofile((Path(target)+'input.txt').abs)
    f = open((Path(target)+'input.txt.hdr').abs,'w')
    f.write(str(out.shape[0]))



@click.group()
def lmrec():
    """
    Interface to LMRec.
    """
    pass


@lmrec.command()
@click.option('--config', '-c', help='Config file', type=click.types.Path(True, dir_okay=False))
@click.option('--target', '-t', help='Target file path', type=click.types.Path(False))
@click.option('--task','-p',help='Recon or Map or both',type=click.Choice(['recon','map','both']))
def preprocess(config,target,task):
    with open(config,'r') as fin:
        c = json.load(fin)          
    gen_script(c,target,task)
    lm2bin(c,target)


@lmrec.command()
@click.option('--scanner','-s',help='Scanner config file',type=click.types.Path(True, dir_okay=False))
@click.option('--task','-t',help='Task file',type=click.types.Path(True, dir_okay=False))
def execute(scanner,task):
    cmd = 'lm-recon '+scanner+' '+task
    os.system(cmd)

@lmrec.command()
@click.option('--source','-s',help='Reconstructed image files',type=click.types.Path(False))
@click.option('--config','-c',help='Config file', type=click.types.Path(True, dir_okay=False))
def postprocess(source,config):
    with open(config,'r') as fin:
        c = json.load(fin)   
    save_result(c,source)
    

if __name__ == "__main__":
    lmrec()

