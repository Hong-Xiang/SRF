import json

from dxl.learn.core import ThisHost

class ImageInfo:
    def __init__(self, grid, center, size):
        self.grid = grid
        self.center = center
        self.size = size


class MapInfo:
    def __init__(self, map_file):
        self._map_file = map_file

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

    def map_file(self, task_index=None):
        if task_index is None:
            task_index = ThisHost.host().task_index
        if isinstance(self._map_file, str):
            return self._map_file
        else:
            return self._map_file[task_index]

    def __str__(self):
        result = {}
        result['map_file'] = self.map_file()
        return json.dumps(result)

class DataInfo:
    def __init__(self,
                 lor_files,
                 lor_shapes,
                 lor_ranges=None,
                 lor_step=None):
        self._lor_files = lor_files
        self._lor_shapes = lor_shapes
        self._lor_ranges = lor_ranges
        self._lor_step = lor_step

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

    def lor_file(self, axis, task_index=None):
        if task_index is None:
            task_index = ThisHost().host().task_index
        if isinstance(self._lor_files[axis], str):
            return self._lor_files[axis]
        else:
            return self._lor_files[axis][task_index]

    def lor_range(self, axis, task_index=None):
        if task_index is None:
            task_index = ThisHost.host().task_index
        if self._lor_ranges is not None:
            return self._maybe_broadcast_value(self._lor_ranges[axis], task_index)
        elif self._lor_step is not None:
            step = self._maybe_broadcast_value(
                self._lor_step[axis], task_index)
            return [task_index * step, (task_index + 1) * step]
        else:
            return None

    def lor_shape(self, axis, task_index=None):
        if task_index is None:
            task_index = ThisHost().host().task_index
        if isinstance(self._lor_shapes[axis], (list, tuple)):
            return self._lor_shapes[axis]
        else:
            return self._lor_shapes[axis][task_index]

    def __str__(self):
        result = {}
        axis = ['x', 'y', 'z']
        result['lor_file'] = {a: self.lor_file(a) for a in axis}
        result['lor_range'] = {a: self.lor_range(a) for a in axis}
        result['lor_shape'] = {a: self.lor_shape(a) for a in axis}
        return json.dumps(result, indent=4, separators=(',', ': '))

class SinoInfo:
    def __init__(self,
                 sino_file,
                 sino_shape):
        self._sino_file = sino_file
        self._sino_shape = sino_shape

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

    def sino_file(self,task_index=None):
        if task_index is None:
            task_index = ThisHost.host().task_index
        if isinstance(self._sino_file,str):
            return self._sino_file
        else:
            return self._sino_file[task_index]

    def sino_shape(self,task_index=None):
        if task_index is None:
            task_index = ThisHost.host().task_index
        if isinstance(self._sino_shape,()):
            return self._sino_shape
        else:
            return self._sino_shape[task_index]
        
    def __str__(self):
        result = {}
        result['sino_file'] = self.sino_file()
        result['sino_shape'] = self.sino_shape()
        return json.dumps(result)

class MatrixInfo:
    def __init__(self,
                 matrix_file,
                 matrix_shape):
        self._matrix_file = matrix_file
        self._matrix_shape = matrix_shape

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

    def matrix_file(self,task_index=None):
        if task_index is None:
            task_index = ThisHost.host().task_index
        if isinstance(self._matrix_file,str):
            return self._matrix_file
        else:
            return self._matrix_file[task_index]

    def matrix_shape(self,task_index=None):
        if task_index is None:
            task_index = ThisHost.host().task_index
        if isinstance(self._matrix_shape,()):
            return self._matrix_shape
        else:
            return self._matrix_shape[task_index]
        
    def __str__(self):
        result = {}
        result['matrix_file'] = self.matrix_file()
        result['matrix_shape'] = self.matrix_shape()
        return json.dumps(result)