import h5py
from jfs.api import Path
import numpy as np
from ..function import get_algorithm

def save_result(config,source): 
    part_config = config['algorithm']['recon']
    nb_subsets,nb_subiterations = get_iter_info(part_config)
    algorithm = get_algorithm(part_config)
    save_path = Path(config['output']['image']['path_file'])
    dataset_prefix = config['output']['image']['path_dataset_prefix']
    if algorithm==('FBP2D'or'FBP3DRP'):
        save_fbp_result(save_path,dataset_prefix,source)
    elif algorithm=='MLEM':
        save_mlem_result(save_path,dataset_prefix,source,nb_subiterations)
    else:
        save_osem_result(save_path,dataset_prefix,source,nb_subiterations,nb_subsets)


def save_fbp_result(save_path,dataset_prefix,source):
    with h5py.File(save_path.abs) as fout:
        label = dataset_prefix
        source_path = Path(source)+'output.v'
        fout[label] = np.fromfile(source_path.abs,dtype='float32')

def save_mlem_result(save_path,dataset_prefix,source,nb_iterations):
    with h5py.File(save_path.abs) as fout:
        for i in range(0,nb_iterations):
            label = dataset_prefix+str(i+1)
            source_path = Path(source)+'output_'+str(i+1)+'.v'
            fout[label] = np.fromfile(source_path.abs,dtype='float32')

def save_osem_result(save_path,dataset_prefix,source,nb_iterations,nb_subsets):   
    with h5py.File(save_path.abs) as fout:
        for i in range(0,nb_iterations):
            for j in range(0,nb_subsets):
                label = dataset_prefix+'output_'+str(i+1)+'_'+str(j+1)
                source_path = Path(source)+str(i+1)+'_'+str(j+1)+'.v'
                fout[label] = np.fromfile(source_path.abs,dtype='float32')

def get_iter_info(config):
    if ("osem" in config):
        nb_subiterations = config['osem']['nb_iterations']
        nb_subsets = config['osem']['nb_subsets']
    else:
        nb_subsets = 1
        nb_subiterations = part_config['mlem']['nb_iterations']
    return nb_subsets,nb_subiterations