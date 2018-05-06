import json

from dxl.learn.core import ThisHost
import h5py
from pathlib import Path
import numpy as np


class ImageInfo:
    def __init__(self, grid: list,
                 center: list,
                 size: list,
                 name: str,
                 map_file: str):
        self.grid = grid
        self.center = center
        self.size = size
        self.name = name
        self.map_file = map_file


class ImageSpec:
    def __init__(self, config):
        self.grid = config['grid']
        self.center = config['center']
        self.size = config['size']
        self.name = config['name']
        self.map_file = config['map_file']

    def to_dict(self):
        return {
            'grid': self.grid,
            'center': self.center,
            'size': self.size,
            'name': self.name,
            'map_file': self.map_file
        }


class OsemInfo:
    def __init__(self,
                 nb_iterations,
                 nb_subsets,
                 save_interval):
        self.nb_iterations = nb_iterations
        self.nb_subsets = nb_subsets
        self.save_interval = save_interval


class OSEMSpec:
    def __init__(self, config):
        self.nb_iterations = config['nb_iterations']
        self.nb_subsets = config['nb_subsets']
        self.save_interval = config['save_interval']

    def to_dict(self):
        return {
            'nb_iterations': self.nb_iterations,
            'nb_subsets': self.nb_subsets,
            'save_interval': self.save_interval
        }


class TorInfo:
    def __init__(self,
                 tof_res,
                 tof_bin):
        self.tof_res = tof_res
        self.tof_bin = tof_bin


class ToFSpec:
    def __init__(self, config: dict):
        self.tof_res = config['tof_res']
        self.tof_bin = config['tof_bin']

    def to_dict(self):
        return {
            'tof_res': self.tof_res,
            'tof_bin': self.tof_bin
        }


class LorsInfo:
    def __init__(self,
                 lors_files,
                 lors_shapes,
                 lors_steps,
                 lors_ranges=None):
        self._lors_files = lors_files
        self._lors_shapes = lors_shapes
        self._lors_ranges = lors_ranges
        self._lors_steps = lors_steps

    def _maybe_broadcast_value(self,
                               value,
                               task_index=None,
                               valid_type=(list, tuple)):
        if task_index is None:
            task_index = ThisHost.host().task_index
        if isinstance(value, valid_type):
            return value[task_index]
        else:
            return value

    def lors_files(self, axis, task_index=None):
        if task_index is None:
            task_index = ThisHost().host().task_index
        # print("lorfiles!!!!!!!!!!!!!!!!!!!!!!!!!:", self._lors_files, type(self._lors_files))
        if isinstance(self._lors_files[axis], str):
            return self._lors_files[axis]
        else:
            return self._lors_files[axis][task_index]

    def lors_ranges(self, axis, task_index=None):
        if task_index is None:
            task_index = ThisHost.host().task_index
        if self._lors_ranges is not None:
            return self._maybe_broadcast_value(self._lors_ranges[axis], task_index)
        elif self._lors_steps is not None:
            step = self._maybe_broadcast_value(
                self._lors_steps[axis], task_index)
            return [task_index * step, (task_index + 1) * step]
        else:
            return None

    def lors_shapes(self, axis, task_index=None):
        if task_index is None:
            task_index = ThisHost().host().task_index
        if isinstance(self._lors_shapes[axis], (list, tuple)):
            return self._lors_shapes[axis]
        else:
            return self._lors_shapes[axis][task_index]

    def lors_steps(self, axis, task_index=None):
        if task_index is None:
            task_index = ThisHost.host().task_index
        return self._lors_steps[axis]

    def __str__(self):
        result = {}
        axis = ['x', 'y', 'z']
        result['lors_file'] = {a: self.lors_files(a) for a in axis}
        result['lors_range'] = {a: self.lors_ranges(a) for a in axis}
        result['lors_shape'] = {a: self.lors_shapes(a) for a in axis}
        return json.dumps(result, indent=4, separators=(',', ': '))


class LoRsSpec:
    def __init__(self, config):
        self.path_file = config['path_file']
        self.path_dataset = config.get('path_dataset', 'lors')
        self._shape = config.get('shapes')
        self._step = config.get('steps')

    def auto_detect(self, nb_workers, nb_subsets):
        p = Path(self.path_file)
        if p.suffix == '.npy':
            lors = np.load(p)
        else:
            raise ValueError(
                "auto_complete for {} not implemented yet.".format(p))
        self._step = lors.shape[0] // (nb_workers * nb_subsets)
        self._shape = [self._step, lors.shape[1]]

    @property
    def shape(self):
        return self._shape

    @property
    def step(self):
        return self._step

    def to_dict(self):
        result = {}
        result['path_file'] = self.path_file
        result['path_dataset'] = self.path_dataset
        if self.shape is not None:
            result['shapes'] = self.shape
        if self.step is not None:
            result['steps'] = self.step
        return result


class LoRsToRSpec(LoRsSpec):
    def __init__(self, config):
        super().__init__(config)

    def auto_complete(self, nb_workers, nb_subsets):
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

    def to_dict(self):
        XYZ = ['x', 'y', 'z']
        result = {}
        result['path_file'] = self.path_file
        result['path_dataset'] = self.path_dataset
        if self.shape is not None:
            result['shapes'] = {a: self.shape[a] for a in XYZ}
        if self.step is not None:
            result['steps'] = {a: self.step[a] for a in XYZ}
        return result


class ToRSpec:
    def __init__(self, config):
        self.kernel_width = config['kernel_width']
        self.c_factor = config.get('c_factor', 0.15)
        self.gaussian_factor = config.get('gaussian_factor', 2.35482005)
        if config.get('preprocess'):
            self.preprocess_lors = LoRsSpec(config['preprocess']['lors'])
        else:
            self.preprocess_lors = None

    def to_dict(self):
        result = {'kernel_width': self.kernel_width}
        if self.c_factor is not None:
            result['c_factor'] = self.c_factor
        if self.gaussian_factor is not None:
            result['gaussian_factor'] = self.gaussian_factor
        if self.preprocess_lors is not None:
            result['preprocess'] = {'lors': self.preprocess_lors.to_dict()}
        return result
