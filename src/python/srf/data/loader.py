import numpy as np
import h5py
from dxl.learn.tensor import constant
from .listmode import ListModeData, ListModeDataSplit,ListModeDataSplitWithoutTOF
from .sinogram_new import SinogramData
from srf.utils.config import config_with_name
from .image import Image
import abc
from srf.io.listmode import load_h5
from srf.preprocess.function.on_tor_lors import str2axis,recon_process

class MasterLoader:
    class KEYS:
        GRID = 'grid'
        CENTER = 'center'
        SIZE = 'size'

    def __init__(self, grid, center, size, name='master_loader'):
        self.config = config_with_name(name)
        self.config.update(self.KEYS.GRID, grid)
        self.config.update(self.KEYS.CENTER, center)
        self.config.update(self.KEYS.SIZE, size)

    def load(self, target_graph):
        return Image(np.ones(self.config[self.KEYS.GRID], dtype=np.float32),
                     self.config[self.KEYS.CENTER],
                     self.config[self.KEYS.SIZE])


class WorkerLoader(abc.ABC):
    class KEYS:
        LORS_PATH = 'lors_path'
        EMAP_PATH = 'emap_path'
        CENTER = 'center'
        SIZE = 'size'

    def __init__(self, lors_path, emap_path, center, size, name='worker_loader'):
        self.config = config_with_name(name)
        self.config.update(self.KEYS.LORS_PATH, lors_path)
        self.config.update(self.KEYS.EMAP_PATH, emap_path)
        self.config.update(self.KEYS.CENTER, center)
        self.config.update(self.KEYS.SIZE, size)

    @abc.abstractmethod
    def load(self, target_graph):
        pass


class SplitWorkerLoader(WorkerLoader):
    def load(self, target_graph):
        if self.config[self.KEYS.LORS_PATH].endswith(".npy"):
            lors = np.load(self.config[self.KEYS.LORS_PATH])
        elif self.config[self.KEYS.LORS_PATH].endswith(".h5"):
            data = load_h5(self.config[self.KEYS.LORS_PATH])
            lors_point = np.hstack((data['fst'],data['snd']))
            lors = np.hstack((lors_point,data['weight'].reshape(data['weight'].size,1)))
        lors = recon_process(lors)
        lors = {k:lors[str2axis(k)] for k in ('x','y','z')}
        projection_data = ListModeDataSplit(
            **{k: ListModeData(lors[k], np.ones([lors[k].shape[0]], np.float32)) for k in lors})
        emap = Image(np.load(self.config[self.KEYS.EMAP_PATH]).astype(np.float32),
                     self.config[self.KEYS.CENTER],
                     self.config[self.KEYS.SIZE])
        return {'projection_data': projection_data, 'efficiency_map': emap}, ()


class CompleteWorkerLoader(WorkerLoader):
    def load(self, target_graph):
        if self.config[self.KEYS.LORS_PATH].endswith(".npy"):
            lors = np.load(self.config[self.KEYS.LORS_PATH])
        elif self.config[self.KEYS.LORS_PATH].endswith(".h5"):
            data = load_h5(self.config[self.KEYS.LORS_PATH])
            lors_point = np.hstack((data['fst'],data['snd']))
            lors = np.hstack((lors_point,data['weight'].reshape(data['weight'].size,1)))
        projection_data = ListModeData(lors, lors[:,6])
        emap = Image(np.load(self.config[self.KEYS.EMAP_PATH]).astype(np.float32),
                     self.config[self.KEYS.CENTER],
                     self.config[self.KEYS.SIZE])
        return {'projection_data': projection_data, 'efficiency_map': emap}, ()


class OSEMWorkerLoader:
    pass

class siddonProjectionLoader(abc.ABC):
    class KEYS:
        LORS_PATH = 'lors_path'
        CENTER = 'center'
        SIZE = 'size'

    def __init__(self, lors_path, center, size, name='siddon_loader'):
        self.config = config_with_name(name)
        self.config.update(self.KEYS.LORS_PATH, lors_path)
        self.config.update(self.KEYS.CENTER, center)
        self.config.update(self.KEYS.SIZE, size)

    def load(self, target_graph):
        if self.config[self.KEYS.LORS_PATH].endswith(".npy"):
            lors = np.load(self.config[self.KEYS.LORS_PATH])
        elif self.config[self.KEYS.LORS_PATH].endswith(".h5"):
            data = load_h5(self.config[self.KEYS.LORS_PATH])
            lors_point = np.hstack((data['fst'],data['snd']))
            lors = np.hstack((lors_point,data['weight'].reshape(data['weight'].size,1)))
        projection_data = ListModeData(lors, np.ones([lors.shape[0]], np.float32))
        return {'projection_data': projection_data}, ()

class siddonSinogramLoader(abc.ABC):
    class KEYS:
        PJ_CONFIG = 'pj_config'
        LORS_PATH = 'lors_path'
        EMAP_PATH = 'emap_path'
        CENTER = 'center'
        SIZE = 'size'

    def __init__(self, pj_config, lors_path, emap_path, center, size, name='sinogram_loader'):
        self.config = config_with_name(name)
        self.config.update(self.KEYS.PJ_CONFIG, pj_config)
        self.config.update(self.KEYS.LORS_PATH, lors_path)
        self.config.update(self.KEYS.EMAP_PATH, emap_path)        
        self.config.update(self.KEYS.CENTER, center)
        self.config.update(self.KEYS.SIZE, size)

    def load(self, target_graph):
        if self.config[self.KEYS.LORS_PATH].endswith(".npy"):
            lors = np.load(self.config[self.KEYS.LORS_PATH])
        elif self.config[self.KEYS.LORS_PATH].endswith(".h5"):
            data = load_h5(self.config[self.KEYS.LORS_PATH])
            lors_point = np.hstack((data['fst'],data['snd']))
            lors = np.hstack((lors_point,data['weight'].reshape(data['weight'].size,1)))
        listmode = ListModeData(lors, np.ones([lors.shape[0]], np.float32))
        # sino = np.load('sinogram_data.npy')
        sino = _listMode2Sino(listmode, self.config[self.KEYS.PJ_CONFIG])
        # np.save('sinogram_data.npy', sino)


        emap = Image(np.load(self.config[self.KEYS.EMAP_PATH]).astype(np.float32),
                self.config[self.KEYS.CENTER],
                self.config[self.KEYS.SIZE])
        return {'projection_data': sino, 'efficiency_map': emap}, ()

def _listMode2Sino(listmode: ListModeData, pj_config) -> SinogramData:
    ang = np.pi * 2 / 16 # config['ring']['nb_blocks_per_ring']
    x_data1 = listmode.lors[:,0]
    y_data1 = listmode.lors[:,1]
    z_data1 = listmode.lors[:,2]
    angles1 = np.arctan2(y_data1, x_data1) + np.pi
    iblock1 = np.round(angles1 / ang) % 16
    angle = iblock1 * ang
    y_data1o = -x_data1 * np.sin(angle) + y_data1 * np.cos(angle)
    y_ind1 = np.round((y_data1o + 33.4 / 2 - 3.34 / 2) / 3.34)
    z_ind1 = np.round((z_data1 + 33.4 / 2 - 3.34 / 2) / 3.34)    

    x_data2 = listmode.lors[:,3]
    y_data2 = listmode.lors[:,4]
    z_data2 = listmode.lors[:,5]
    angles2 = np.arctan2(y_data2, x_data2) + np.pi
    iblock2 = np.round(angles2 / ang) % 16
    angle = iblock2 * ang
    y_data2o = -x_data2 * np.sin(angle) + y_data2 * np.cos(angle)
    y_ind2 = np.round((y_data2o + 33.4 / 2 - 3.34 / 2) / 3.34)
    z_ind2 = np.round((z_data2 + 33.4 / 2 - 3.34 / 2) / 3.34) 

    sino = np.zeros((10*10*16, 10*10*16), dtype = int)
    print(listmode.lors[0,:])
    ind1 = (y_ind1 + 10 * z_ind1 + 100 * iblock1).astype(int)
    ind2 = (y_ind2 + 10 * z_ind2 + 100 * iblock2).astype(int)

    for i in range(len(x_data1)):
        if i * 100 % len(x_data1) == 0:
            print(i * 100 / len(x_data1))
        sino[ind1[i], ind2[i]] = sino[ind1[i], ind2[i]] + 1
        sino[ind2[i], ind1[i]] = sino[ind2[i], ind1[i]] + 1

    return SinogramData(sino)