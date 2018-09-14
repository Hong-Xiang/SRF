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


class ReconSpec(Specs):
    FIELDS = ('nb_iterations', 'nb_subsets'  'save_interval')



class SinoSpec(Specs):
    FIELDS = ('path_file', 'path_dataset', 'slices', 'shape')





class SRFTaskSpec(Specs):
    FIELDS = ('work_directory', 'task_type')
    TASK_TYPE = None

    def __init__(self, config):
        super().__init__(config)
        self.data['task_type'] = self.TASK_TYPE


class SinoTaskSpec(SRFTaskSpec):
    # from ..graph.pet.tor import TorReconstructionTask
    TASK_TYPE = 'SinoTask'
    # task_cls = TorReconstructionTask

    class KEYS:
        IMAGE = 'image'
        SINO = 'sino'
        MATRIX = 'matrix'
        RECON = 'recon'


    FIELDS = tuple(list(SRFTaskSpec.FIELDS) + [KEYS.IMAGE, KEYS.SINO,
                                               KEYS.MATRIX, KEYS.RECON])

    def __init__(self, config):
        super().__init__(config)
        
    def parse(self, key, cls):
        if key in self.data:
            self.data[key] = cls(self.data[key])

