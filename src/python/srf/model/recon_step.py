from dxl.learn import Model
from srf.data import Image
from doufo import func, multidispatch

"""
ReconstructionStep is the abstract representation of 
'one step of medical image reconstruction',
It currently representing:
1. projection = Projection(Image, ProjectionDomain)
2. backprojection = Backprojection(projection::ProjectionData, ImageDomain)
3. image_next = image / efficiency_map * backprojection
"""

__all__ = ['ReconStep', 'mlem_update', 'mlem_update_normal']


class ReconStep(Model):
    class KEYS(Model.KEYS):
        class TENSOR:
            IMAGE = 'image'
            EFFICIENCY_MAP = 'efficiency_map'
            PROJECTION_DATA = 'projection_data'

    def __init__(self, name, projection, backprojection, update):
        super().__init__(name)
        self.projection = projection
        self.backprojection = backprojection
        self.update = update

    def build(self, *args):
        pass

    def kernel(self, inputs):
        image = inputs[self.KEYS.TENSOR.IMAGE]
        efficiency_map = inputs[self.KEYS.TENSOR.EFFICIENCY_MAP]
        projection_data = inputs[self.KEYS.TENSOR.PROJECTION_DATA]
        proj = self.projection(image, projection_data)
        back_proj = self.backprojection(proj, image)
        return self.update(image, back_proj, efficiency_map)

    @property
    def parameters(self):
        return []


def mlem_update(image_prev: Image, image_succ: Image, efficiency_map: Image):
    return image_prev.fmap(lambda d: d * efficiency_map.data * image_succ.data)


def mlem_update_normal(image_prev: Image, image_succ: Image, efficiency_map: Image):
    return image_prev.fmap(lambda d: d / efficiency_map.data * image_succ.data)

