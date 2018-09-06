import numpy as np
import h5py
from dxl.learn.tensor import constant
from .listmode import ListModeData
from srf.utils.config import config_with_name
from .image import Image


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


class SplitWorkerLoader:
    def __init__(self, lors_path, emap_path):
        self.lors_path = lors_path
        self.emap_path = emap_path

    def load(self, target_graph):
        # lors = load_npz(self.lors_path)
        # lors = {
        #     a: Constant(lors[a].astype(np.float32), 'lors_{}'.format(a))
        #     for a in ['x', 'y', 'z']
        # }
        lors = self.lors_loader.load()
        emap = np.load(self.emap_path).astype(np.float32)
        return {'projection_data': lors, 'efficiency_map': emap}, ()


class CompleteWorkerLoader:
    class KEYS:
        LORS_PATH = 'lors_path'
        EMAP_PATH = 'emap_path'
        CENTER = 'center'
        SIZE = 'size'

    def __init__(self, lors_path, emap_path, center, size, name='complete_worker_loader'):
        self.config = config_with_name(name)
        self.config.update(self.KEYS.LORS_PATH, lors_path)
        self.config.update(self.KEYS.EMAP_PATH, emap_path)
        self.config.update(self.KEYS.CENTER, center)
        self.config.update(self.KEYS.SIZE, size)

    def load(self, target_graph):
        # with h5py.File(self.config[self.KEYS.LORS_PATH]) as fin:
        #     fst = np.array(fin['listmode_data']['fst'], np.float32)
        #     snd = np.array(fin['listmode_data']['snd'], np.float32)
        #     weights = np.array(fin['listmode_data']['weight'], np.float32)
        #     tof = np.zeros_like(weights).reshape([-1, 1])
        #     lors = np.concatenate([fst, snd, tof], axis=1)
        #     projection_data = ListModeData(lors, weights)

        lors = np.load(self.config[self.KEYS.LORS_PATH])
        projection_data = ListModeData(lors, np.ones([lors.shape[0]], np.float32))
        emap = Image(np.load(self.config[self.KEYS.EMAP_PATH]).astype(np.float32),
                     self.config[self.KEYS.CENTER],
                     self.config[self.KEYS.SIZE])
        return {'projection_data': projection_data, 'efficiency_map': emap}, ()


class OSEMWorkerLoader:
    pass
