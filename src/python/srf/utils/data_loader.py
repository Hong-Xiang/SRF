from pathlib import Path
import numpy as np


class LocalDataLoader:
    def load(self, config):
        return self.load_data_kernel(**self.data_spec[key])


class MasterLoader:
    def load(self, config):
        return np.ones(config('image')['shape'])
