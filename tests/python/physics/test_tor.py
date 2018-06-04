from srf.physics import ToRModel
from srf.tensor import Image
from srf.test import TestCase
from dxl.learn.core import Constant
import numpy as np
import json


class TestToRModel(TestCase):
    def get_data(self):
        path = self.resource_path / 'model' / 'projection' / 'ToR'
        image = np.load(path / 'test_image.npy').astype(np.float32)
        with open(path / 'config.json') as fin:
            config = json.load(fin)
        image = Constant(image, 'image')
        assert tuple(image.shape) == tuple(config['grid'])
        image = Image(image, config['center'], config['size'])
        lors = np.load(path / 'test_lors.npy').astype(np.float32)
        lors = Constant(lors, 'lors')
        expected = np.load(path / 'test_projection.npy')
        return image, lors, expected

    def get_model(self):
        path = self.resource_path / 'model' / 'projection' / 'ToR'
        with open(path / 'config.json') as fin:
            config = json.load(fin)
        return ToRModel(name='model', kernel_width=config['kernel_width'],
                        tof_bin=config['tof_bin'], tof_sigma2=config['tof_sigma2'])

    def test_calculate(self):
        image, lors, expected = self.get_data()
        model = self.get_model()
        proj = model.projection(image, lors)
        with self.test_session() as sess:
            result = sess.run(proj)
        self.assertFloatArrayEqual(
            expected, result, "Projection data not corrected.")
