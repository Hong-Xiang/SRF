import json
import scipy.io as sio

from dxl.learn.core import ThisHost
import h5py
from pathlib import Path
import numpy as np
from ..preprocess import preprocess_sino


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
        self.path_file = config['path_file']
        self.path_dataset = config['path_dataset']
        self.map_file = config['map']['path_file']
        self.map_dataset = config['map']['path_dataset']

    def to_dict(self):
        return {
            'grid': [self.grid[0]*self.grid[1]*self.grid[2],1],
            'path_file': self.path_file,
            'path_dataset':self.path_dataset,
            'map_file': self.map_file,
            'map_dataset': self.map_dataset
        }


class ReconInfo:
    def __init__(self,
                 nb_iterations,
                 save_interval):
        #self.algorithm = algorithm
        self.nb_iterations = nb_iterations
        #self.nb_subsets = nb_subsets
        self.save_interval = save_interval

class ReconSpec:
    def __init__(self, config):
        self.nb_iterations = config['nb_iterations']
        #self.nb_subsets = config['nb_subsets']
        self.save_interval = config['save_interval']

    def to_dict(self):
        return {
            'nb_iterations': self.nb_iterations,
            'save_interval': self.save_interval
        }



class InputInfo:
    def __init__(self,
                 Input_file,
                 sm):
        #self.datatype = datatype
        self.Input_file = Input_file
        self.sm = sm



class SinoInfo:
    def __init__(self,
                 sino_file,
                 sino_shape,
                 sino_steps,
                 sino_range=None):
        self._sino_file = sino_file
        self._sino_shape = sino_shape
        self._sino_steps = sino_steps
        self._sino_range = sino_range

    def _maybe_broadcast_value(self,
                               value,
                               task_index=None,
                               valid_type=(list, tuple)):
        if task_index is None:
            task_index = ThisHost
        if isinstance(value, valid_type):
            return value[task_index]
        else:
            return value

    def sino_file(self,  task_index=None):
        if task_index is None:
            task_index = ThisHost().host().task_index
        if isinstance(self._sino_file, str):
            return self._sino_file
        else:
            return self._sino_file[task_index]

    def sino_range(self,  task_index=None):
        if task_index is None:
            task_index = ThisHost.host().task_index
        if self._sino_range is not None:
            return self._maybe_broadcast_value(self._sino_range, task_index)
        else:
            return None

    def sino_shape(self, task_index=None):
        if task_index is None:
            task_index = ThisHost().host().task_index
        if isinstance(self._sino_shape, (list, tuple)):
            return self._sino_shape
        else:
            return self._sino_shape[task_index]

    def sino_steps(self, task_index = None):
        if task_index is  None:
            task_index = ThisHost.host().task_index
        return self._sino_steps

    def __str__(self):
        result = {}
        result['sino_file'] =  self.sino_file()
        result['sino_range'] = self.sino_range()
        result['sino_shape'] = self.sino_shape()
        return json.dumps(result, indent=3, separators=(',', ': '))


class SinoSpec:
    def __init__(self, config):
        self.path_file = config['path_file']
        self.path_dataset = config['path_dataset']
        # self.path_dataset = config.get('path_dataset', 'sino')
        self._shape = config.get('shapes')
        self._step = config.get('steps')


    def auto_detect(self, nb_workers):
        p = Path(self.path_file)
        if p.suffix == '.npy':
            sino = np.load(p)
        else:
            raise ValueError(
                "auto_complete for {} not implemented yet.".format(p))
        sino = preprocess_sino.preprocess_sino(sino)
        self._step = sino.shape[0] // (nb_workers)
        self._shape = [self._step, sino.shape[1]]

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

class SinoToRSpec(SinoSpec):
    def __init__(self, config):
        super().__init__(config)

    def auto_complete(self, nb_workers):
        """
        Complete infomation with nb_workes given.

        If ranges is None, infer by steps [i*step, (i+1)*step].
        If step is None, infer by shape
        """
        with h5py.File(self.path_file) as fin:
            sino = fin[self.path_dataset]
            self._steps =  sino.shape[0] //(nb_workers) 
            self._shapes =  [self._steps, sino.shape[1]]
        

    def _maybe_broadcast_ints(self, value, task_index):
        if task_index is None:
            task_index = ThisHost().host().task_index
        else:
            task_index = int(task_index)
        if len(value) <= task_index or isinstance(value[task_index], int):
            return value
        return value[task_index]

    def sino_shapes(self,  task_index=None):
        return self._maybe_broadcast_ints(self._shapes, task_index)

    def sino_steps(self, task_index=None):
        return self._maybe_broadcast_ints(self._steps, task_index)

    def to_dict(self):
        # XYZ = ['x', 'y', 'z']
        result = {}
        result['path_file'] = self.path_file
        result['path_dataset'] = self.path_dataset
        if self.shape is not None:
            result['shapes'] = self.shape
        if self.step is not None:
            result['steps'] = self.step
        return result


class MatrixInfo:
    def __init__(self,
                 matrix_file,
                 matrix_shape,
                 matrix_steps):
        self._matrix_file = matrix_file
        self._matrix_shape = matrix_shape
        self._matrix_steps = matrix_steps

    def _maybe_broadcast_value(self,
                               value,
                               task_index=None,
                               valid_type=(list, tuple)):
        if task_index is None:
            task_index = ThisHost
        if isinstance(value, valid_type):
            return value[task_index]
        else:
            return value

    def matrix_file(self,  task_index=None):
        if task_index is None:
            task_index = ThisHost().host().task_index
        if isinstance(self._matrix_file, str):
            return self._matrix_file
        else:
            return self._matrix_file[task_index]

    # def sino_range(self,  task_index=None):
    #     if task_index is None:
    #         task_index = ThisHost.host().task_index
    #     if self._sino_range is not None:
    #         return self._maybe_broadcast_value(self._sino_range, task_index)
    #     else:
    #         return None

    def matrix_shape(self, task_index=None):
        if task_index is None:
            task_index = ThisHost().host().task_index
        if isinstance(self._matrix_shape, (list, tuple)):
            return self._matrix_shape
        else:
            return self._matrix_shape[task_index]

    def matrix_steps(self, task_index = None):
        if task_index is  None:
            task_index = ThisHost.host().task_index
        return self._matrix_steps

    def __str__(self):
        result = {}
        result['matrix_file'] =  self.matrix_file
        result['matrix_range'] = self.matrix_range
        result['matrix_shape'] = self.matrix_shape
        return json.dumps(result, indent=3, separators=(',', ': '))


class MatrixSpec:
    def __init__(self, config):
        self.path_file = config['path_file']
        self.path_dataset = config.get('path_dataset', 'matrix')
        self._shape = config.get('shapes')
        self._step = config.get('steps')

    def auto_detect(self, nb_workers):
        p = Path(self.path_file)
        if p.suffix == '.npy':
            matrix = np.load(p)
        else:
            raise ValueError(
                "auto_complete for {} not implemented yet.".format(p))
        self._step = matrix.shape[0] // (nb_workers)
        self._shape = [self._step, matrix.shape[1]]

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

class MatrixToRSpec(MatrixSpec):
    def __init__(self, config):
        super().__init__(config)

    def auto_complete(self, nb_workers):
        """
        Complete infomation with nb_workes given.

        If ranges is None, infer by steps [i*step, (i+1)*step].
        If step is None, infer by shape
        """
        # with h5py.File(self.path_file) as fin:
        #     matrix = fin[self.path_dataset]
        #     self._steps = {a: v.shape[0] //
        #                    (nb_workers) for a, v in matrix.items()}
        #     self._shapes = {a: [self._steps[a], v.shape[1]]
        #                     for a, v in matrix.items()}
        fin = sio.loadmat(self.path_file)
        matrix = fin[self.path_dataset]
        self._steps = matrix.shape[0]//nb_workers
        self._shape = [self._steps,matrix.shape[1]]

    def _maybe_broadcast_ints(self, value, task_index):
        if task_index is None:
            task_index = ThisHost().host().task_index
        else:
            task_index = int(task_index)
        if len(value) <= task_index or isinstance(value[task_index], int):
            return value
        return value[task_index]

    def matrix_shapes(self,  task_index=None):
        return self._maybe_broadcast_ints(self._shapes, task_index)

    def matrix_steps(self, task_index=None):
        return self._maybe_broadcast_ints(self._steps, task_index)

    def to_dict(self):
        # XYZ = ['x', 'y', 'z']
        result = {}
        result['path_file'] = self.path_file
        result['path_dataset'] = self.path_dataset
        if self.shape is not None:
            result['shapes'] = self.shape
        if self.step is not None:
            result['steps'] = self.step
        return result