from dxl.learn import Model
from srf.data import Image
from doufo import func, multidispatch


__all__ = ['ProjStep']


class ProjStep(Model):
    class KEYS(Model.KEYS):
        class TENSOR:
            IMAGE_DATA = 'image_data'
            PROJECTION = 'projection'

    def __init__(self, name, projection):
        super().__init__(name)
        self.projection = projection

    def build(self, *args):
        pass

    def kernel(self, inputs):
        image_data = inputs[self.KEYS.TENSOR.IMAGE_DATA]
        proj = inputs[self.KEYS.TENSOR.PROJECTION]
        proj = self.projection(image_data, proj)
        return proj

    @property
    def parameters(self):
        return []

