from dxl.learn import Model
from srf.data import Image
from doufo import func, multidispatch
import numpy as np

__all__ = ['ProjStep']


class ProjStep(Model):
    class KEYS(Model.KEYS):
        class TENSOR:
            IMAGE = 'image'
            PROJECTION_DATA = 'projection_data'

    def __init__(self, name, projection):
        super().__init__(name)
        self.projection = projection

    def build(self, *args):
        pass

    def kernel(self, inputs):
        image_data = inputs[self.KEYS.TENSOR.IMAGE]
        image = Image(image_data, [0.0, 0.0, 0.0], [220.0, 220.0, 30.0])
        projection_data = inputs[self.KEYS.TENSOR.PROJECTION_DATA]
        proj = self.projection(image, projection_data)
        return proj

    @property
    def parameters(self):
        return []
