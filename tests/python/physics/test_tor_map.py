
import numpy as np
import json
from srf.test import TestCase 
from srf.tensor import Image
from dxl.learn.core import Constant

class TestToRMapModel(TestCase):
    def get_data(self):
        path = self.resource_path / 'physics' / 'ToRMap'
        image = np.load(path / 'test_image.npy').astype(np.float32)
        with open(path / 'config.json') as fin:
            config = json.load(fin)
        image = Constant(image, 'image')
        assert tuple(image.shape) == tuple(config['grid'])
        image = Image(image, config['center'], config['size'])
        lors = np.load(path / 'test_lors.npy')