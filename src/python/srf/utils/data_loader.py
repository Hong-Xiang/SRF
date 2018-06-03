from pathlib import Path


class LocalDataLoader:
    def __init__(self, data_spec):
        self.data_spec = data_spec

    def load(self, key):
        return self.load_data_kernel(**self.data_spec[key])

    def load_data_kernel(self, path_file, path_dataset=None, slices=None):
        pass
