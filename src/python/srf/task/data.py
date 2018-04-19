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


class OsemInfo:
    def __init__(self,
                 nb_iterations,
                 nb_subsets,
                 save_interval):
        self.nb_iterations = nb_iterations
        self.nb_subsets = nb_subsets
        self.save_interval = save_interval


class TorInfo:
    def __init__(self,
                 tof_res,
                 tof_bin):
        self.tof_res = tof_res
        self.tof_bin = tof_bin


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
            task_index = ThisHost
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

    def lors_steps(self, axis, task_index = None):
        if task_index is  None:
            task_index = ThisHost.host().task_index
        return self._lors_steps[axis]


    def __str__(self):
        result = {}
        axis = ['x', 'y', 'z']
        result['lors_file'] = {a: self.lors_files(a) for a in axis}
        result['lors_range'] = {a: self.lors_ranges(a) for a in axis}
        result['lors_shape'] = {a: self.lors_shapes(a) for a in axis}
        return json.dumps(result, indent=4, separators=(',', ': '))
