import numpy as np
import h5py
from dxl.learn.tensor import constant
from .listmode import ListModeData, ListModeDataSplit,ListModeDataSplitWithoutTOF
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
        projection_data = ListModeData(lors, np.ones([lors.shape[0]], np.float32))
        emap = Image(np.load(self.config[self.KEYS.EMAP_PATH]).astype(np.float32),
                     self.config[self.KEYS.CENTER],
                     self.config[self.KEYS.SIZE])
        return {'projection_data': projection_data, 'efficiency_map': emap}, ()


class OSEMWorkerLoader:
    pass
