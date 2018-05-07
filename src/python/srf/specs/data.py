import json

from dxl.learn.core import ThisHost
import h5py
from pathlib import Path
import numpy as np

from collections import UserDict


class Specs(UserDict):
    FIELDS = tuple()

    def __init__(self, config):
        self.data = {k: v for k, v in config.items()
                     if k in self.FIELDS}

    def __getattr__(self, key):
        if key in self.FIELDS:
            return self.data[key]
        raise KeyError("Key {} not found.".format(key))


class NDArraySpec(Specs):
    FIELDS = ('path_file', 'path_dataset', 'slices')


class ImageSpec(Specs):
    FIELDS = ('grid', 'center', 'size', 'name', 'map_file')

    # def __init__(self, config):
    #     self.data = config
    #     self.grid = config['grid']
    #     self.center = config['center']
    #     self.size = config['size']
    #     self.name = config['name']
    #     self.map_file = config['map_file']


class OSEMSpec(Specs):
    FIELDS = ('nb_iterations', 'nb_subsets', 'save_interval')


class ToFSpec(Specs):
    FIELDS = ('tof_res', 'tof_bin')


class LoRsSpec(Specs):
    FIELDS = ('path_file', 'path_dataset', 'slices', 'shape')

    # def __init__(self, config):
    # super().__init__(config)
    # self._shape = None

    # def __init__(self, config):
    # self.path_file = config['path_file']
    # self.path_dataset = config.get('path_dataset', 'lors')
    # self._shape = config.get('shapes')
    # self._step = config.get('steps')

    # def auto_detect(self, nb_workers, nb_subsets):
    #     p = Path(self.path_file)
    #     if p.suffix == '.npy':
    #         lors = np.load(p)
    #     else:
    #         raise ValueError(
    #             "auto_complete for {} not implemented yet.".format(p))
    #     self._step = lors.shape[0] // (nb_workers * nb_subsets)
    #     self._shape = [self._step, lors.shape[1]]

    # @property
    # def shape(self):
    #     if self._shape is not None:
    #         return self._shape
    #     elif self.data.get('slices') is not None:
    #         slieces = self.data.get('slices')
    #         if isinstance(slieces, str):
    #             from dxl.data.utils.slices import slices_from_str
    #             slices = slices_from_str(slices)
    #         self.shape = tuple([s.])

    #     return self._shape

    # @property
    # def step(self):
    #     return self._step

    # def to_dict(self):
    #     result = {}
    #     result['path_file'] = self.path_file
    #     result['path_dataset'] = self.path_dataset
    #     if self.shape is not None:
    #         result['shapes'] = self.shape
    #     if self.step is not None:
    #         result['steps'] = self.step
    #     return result


class LoRsToRSpec(LoRsSpec):

    def auto_complete(self, nb_workers, nb_subsets=1):
        """
        Complete infomation with nb_workes given.

        If ranges is None, infer by steps [i*step, (i+1)*step].
        If step is None, infer by shape
        """
        with h5py.File(self.path_file) as fin:
            lors3 = fin[self.path_dataset]
            self._steps = {a: v.shape[0] //
                           (nb_workers * nb_subsets) for a, v in lors3.items()}
            self._shapes = {a: [self._steps[a], v.shape[1]]
                            for a, v in lors3.items()}

    def _maybe_broadcast_ints(self, value, task_index):
        if task_index is None:
            task_index = ThisHost().host().task_index
        else:
            task_index = int(task_index)
        if len(value) <= task_index or isinstance(value[task_index], int):
            return value
        return value[task_index]

    def lors_shapes(self, axis, task_index=None):
        return self._maybe_broadcast_ints(self._shapes[axis], task_index)

    def lors_steps(self, axis, task_index=None):
        return self._maybe_broadcast_ints(self._steps[axis], task_index)

    # def to_dict(self):
    #     XYZ = ['x', 'y', 'z']
    #     result = {}
    #     result['path_file'] = self.path_file
    #     result['path_dataset'] = self.path_dataset
    #     if self.shape is not None:
    #         result['shapes'] = {a: self.shape[a] for a in XYZ}
    #     if self.step is not None:
    #         result['steps'] = {a: self.step[a] for a in XYZ}
    #     return result


class ToRSpec(Specs):
    class KEYS:
        PREPROCESS_LORS = 'preprocess_lors'
    FIELDS = ('kernel_width', 'gaussian_factor',
              'c_factor', KEYS.PREPROCESS_LORS)

    def __init__(self, config):
        super().__init__(config)
        if self.KEYS.PREPROCESS_LORS in self.data:
            self.data[self.KEYS.PREPROCESS_LORS] = LoRsSpec(
                self.data[self.KEYS.PREPROCESS_LORS])


class SRFTaskSpec(Specs):
    FIELDS = ('work_directory', 'task_type')
    TASK_TYPE = None

    def __init__(self, config):
        super().__init__(config)
        self.data['task_type'] = self.TASK_TYPE


class ToRTaskSpec(SRFTaskSpec):
    # from ..graph.pet.tor import TorReconstructionTask
    TASK_TYPE = 'TorTask'
    # task_cls = TorReconstructionTask

    class KEYS:
        IMAGE = 'image'
        LORS = 'lors'
        TOF = 'tof'
        OSEM = 'osem'
        TOR = 'tor'

    FIELDS = tuple(list(SRFTaskSpec.FIELDS) + [KEYS.IMAGE, KEYS.LORS,
                                               KEYS.TOF, KEYS.TOR, KEYS.OSEM])

    def __init__(self, config):
        super().__init__(config)
        # self.parse(self.KEYS.IMAGE, ImageSpec)
        # self.parse(self.KEYS.LORS, LoRsToRSpec)
        # self.parse(self.KEYS.TOF, ToFSpec)
        # self.parse(self.KEYS.OSEM, OSEMSpec)
        # self.parse(self.KEYS.TOR, ToRSpec)

        # self.image = ImageSpec(config['image'])
        # self.lors = LoRsToRSpec(config['lors'])
        # self.tof = ToFSpec(config['tof'])
        # self.osem = OSEMSpec(config['osem'])
        # self.tor = ToRSpec(config['tor'])

    def parse(self, key, cls):
        if key in self.data:
            self.data[key] = cls(self.data[key])

    # def to_dict(self):
    #     result = super().to_dict()
    #     result.update({
    #         'image': self.image.to_dict(),
    #         'lors': self.lors.to_dict(),
    #         'tof': self.tof.to_dict(),
    #         'osem': self.osem.to_dict(),
    #         'tor': self.tor.to_dict()
    #     })
    #     return result
    # #     self.kernel_width = config['kernel_width']
    #     self.c_factor = config.get('c_factor', 0.15)
    #     self.gaussian_factor = config.get('gaussian_factor', 2.35482005)
    #     if config.get('preprocess'):
    #         self.preprocess_lors = LoRsSpec(config['preprocess']['lors'])
    #     else:
    #         self.preprocess_lors = None

    # def to_dict(self):
    #     result = {'kernel_width': self.kernel_width}
    #     if self.c_factor is not None:
    #         result['c_factor'] = self.c_factor
    #     if self.gaussian_factor is not None:
    #         result['gaussian_factor'] = self.gaussian_factor
    #     if self.preprocess_lors is not None:
    #         result['preprocess'] = {'lors': self.preprocess_lors.to_dict()}
    #     return result
