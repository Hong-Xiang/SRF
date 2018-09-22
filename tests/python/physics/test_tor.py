# from srf.physics import ToRModel
from srf.tensor import Image
from srf.test import TestCase
from dxl.learn.core import Constant
import numpy as np
import json
import pytest

class TestToRModel(TestCase):
    def get_data(self):
        path = self.resource_path / 'physics' / 'ToR'
        image = np.load(path / 'test_image.npy').astype(np.float32)
        with open(path / 'config.json') as fin:
            config = json.load(fin)
        image = Constant(image, 'image')
        assert tuple(image.shape) == tuple(config['grid'])
        image = Image(image, config['center'], config['size'])
        lors = np.load(path / 'test_lors.npy').astype(np.float32)
        lors = Constant(lors, 'lors')
        projected = np.load(path / 'test_projection.npy')
        back_projected = np.load(path / 'test_backprojection.npy')
        lors_value = Constant(projected, 'lors_value')
        return {
            'image': image,
            'lors': lors,
            'lors_value': lors_value,
            'projected': projected,
            'backprojected': back_projected
        }

    def get_model(self):
        path = self.resource_path / 'physics' / 'ToR'
        with open(path / 'config.json') as fin:
            config = json.load(fin)
        return ToRModel(name='model', kernel_width=config['kernel_width'],
                        tof_bin=config['tof_bin'], tof_sigma2=config['tof_sigma2'])

    @pytest.mark.skip(reason='FIX ME')
    def test_projection(self):
        data = self.get_data()
        image, lors, expected = data['image'], data['lors'], data['projected']
        model = self.get_model()
        proj = model.projection(image, lors)
        with self.test_session() as sess:
            result = sess.run(proj)
        self.assertFloatArrayEqual(
            expected, result, "Projection data not corrected.")
    @pytest.mark.skip(reason='FIX ME')
    def test_backprojection(self):
        data = self.get_data()
        lors, lors_value, image, expected = data['lors'], data[
            'lors_value'], data['image'], data['backprojected']
        model = self.get_model()
        back_proj = model.backprojection(
            {'lors': lors, 'lors_value': lors_value}, image)
        with self.test_session() as sess:
            result = sess.run(back_proj)
        result = result
        self.assertFloatArrayEqual(
            expected, result, "Backrojection data not corrected.")
