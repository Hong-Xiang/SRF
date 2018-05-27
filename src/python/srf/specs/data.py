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

    def parse(self, key, cls):
        if key in self.data:
            self.data[key] = cls(self.data[key])
