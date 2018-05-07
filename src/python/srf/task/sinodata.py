import json

from dxl.learn.core import ThisHost


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


class ReconInfo:
    def __init__(self,
                 nb_iterations,
                 save_interval):
        #self.algorithm = algorithm
        self.nb_iterations = nb_iterations
        #self.nb_subsets = nb_subsets
        self.save_interval = save_interval



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
