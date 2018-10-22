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

__all__ = ['BprojStep']


class BprojStep(Model):
    class KEYS(Model.KEYS):
        class TENSOR:
            IMAGE = 'image'
            PROJECTION_DATA = 'projection_data'

    def __init__(self, name, backprojection):
        super().__init__(name)
        self.backprojection = backprojection

    def build(self, *args):
        pass

    def kernel(self, inputs):
        image = inputs[self.KEYS.TENSOR.IMAGE]
        projection_data = inputs[self.KEYS.TENSOR.PROJECTION_DATA]
        back_proj = self.backprojection(projection_data, image)
        return back_proj

    @property
    def parameters(self):
        return []

