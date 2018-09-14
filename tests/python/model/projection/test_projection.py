from srf.model import ProjectionToR
from srf.tensor import Image
from srf.test import TestCase
from dxl.learn.core import Constant
import numpy as np
import json


class TestProjectionTor(TestCase):
    def get_data(self):
        path = self.resource_path() / 'model' / 'projection' / 'ToR'
        image = np.load(path / 'test_image.npy')
        with open(path / 'config.json') as fin:
            config = json.load(fin)
        image = Constant(image, 'image')
        assert tuple(image.shape) == tuple(config['grid'])
        image = Image(image, config['center'], config['size'])
