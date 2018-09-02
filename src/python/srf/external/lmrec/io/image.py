from jfs.api import Path
import numpy as np
import h5py

def get_iter_info(config):
    if ("osem" in config):
        nb_subiterations = config['osem']['nb_iterations']
    else:
        nb_subiterations = part_config['mlem']['nb_iterations']
    return nb_subiterations

def save_h5_data(path,dataset,source,nb_iterations):
    with h5py.File(path.abs) as fout:
        for i in range(0,nb_iterations):
            label = dataset+str(i+1)
            source_path = Path(source)+'output_'+str(i+1)+'.rec'
            fout[label] = np.fromfile(source_path.abs,dtype='float32')

def save_result(config,source): 
    part_config = config['algorithm']['recon']
    nb_subiterations = get_iter_info(part_config)
    save_path = Path(config['output']['image']['path_file'])
    dataset_prefix = config['output']['image']['path_dataset_prefix']
    save_h5_data(save_path,dataset_prefix,source,nb_subiterations)


