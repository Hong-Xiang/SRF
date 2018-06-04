from dxl.learn.core import ConfigurableWithName, Constant
import numpy as np
from dxl.data.io import load_npz


class MasterLoader:
    def __init__(self, shape):
        self.shape = shape

    def load(self, target_graph):
        # return Constant(np.ones(self.shape, dtype=np.float32), 'x_init')
        return np.ones(self.shape, dtype=np.float32)


class WorkerLoader:
    def __init__(self):
        pass

    def load(self, target_graph):
        lors = load_npz('lors_tor.npz')
        lors = {
            a: Constant(lors[a], 'lors_{}'.format(a))
            for a in ['x', 'y', 'z']
        }
        emap = np.load('map.npy')
        emap = Constant(emap, 'emap')
        return {'projection_data': lors, 'efficiency_map': emap}, ()


class OSEMWorkerLoader:
    pass
