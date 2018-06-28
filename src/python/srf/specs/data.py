import json

from dxl.learn.distribute import ThisHost
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


class OSEMSpec(Specs):
    FIELDS = ('nb_iterations', 'nb_subsets', 'save_interval')


class ToFSpec(Specs):
    FIELDS = ('tof_res', 'tof_bin')


class LoRsSpec(Specs):
    FIELDS = ('path_file', 'path_dataset', 'slices', 'shape')


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
