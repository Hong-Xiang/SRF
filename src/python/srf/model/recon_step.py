from dxl.learn.core import Model
from dxl.learn.core import SubgraphPartialMaker
from srf.tensor import Image

"""
ReconstructionStep is the abstract representation of 'one step of medical image reconstruction',
It currently representing:
1. projection = Projection(Image, ProjectionDomain)
2. backprojection = Backprojection(projection::ProjectionData, ImageDomain)
3. image_next = image / efficiency_map * backprojection
"""
from dxl.data import func

class ReconStep(Model):
    class KEYS(Model.KEYS):
        class TENSOR(Model.KEYS.TENSOR):
            IMAGE = 'image'
            EFFICIENCY_MAP = 'efficiency_map'
            PROJECTION_DATA = 'projection_data'

        class GRAPH(Model.KEYS.GRAPH):
            PROJECTION = 'projection'
            BACKPROJECTION = 'backprojection'

    def __init__(self, info, projection, backprojection):
        super().__init__(info, graphs={
            self.KEYS.GRAPH.PROJECTION: projection,
            self.KEYS.GRAPH.BACKPROJECTION: backprojection
        })

    def kernel(self, inputs):
        KT, KS = self.KEYS.TENSOR, self.KEYS.GRAPH
        image, proj_data = inputs[KT.IMAGE], inputs[KT.PROJECTION_DATA]
        image = Image(image, self.config('center'), self.config('size'))
        proj = self.graphs[KS.PROJECTION](
            {'image': image, 'projection_data': proj_data})
        back_proj = self.graphs[KS.BACKPROJECTION]({
            'projection_data': {'lors': proj_data, 'lors_value': proj},
            'image': image
        })
        result = image / inputs[KT.EFFICIENCY_MAP] * back_proj
        return result

@func
def mlem_update(image_prev, image_succ):
    return image_prev * image_succ

@func
def normalize_efficienncy_map(efficiency_map, image):
    return image / efficiency_map

# class ReconStepHardCoded(ReconStep):
#     def __init__(self, info, *, inputs, config=None):
#         super().__init__(info, inputs=inputs, config=config)

#     def kernel(self, inputs):
#         KT, KS = self.KEYS.TENSOR, self.KEYS.GRAPH
#         image, proj_data = inputs[KT.IMAGE], inputs[KT.PROJECTION_DATA]
#         from ..physics import ToRModel
#         from .projection import ProjectionToR
#         from .backprojection import BackProjectionToR
#         pm = ToRModel('projection_model')
#         proj = ProjectionToR(self.info.child_scope(
#             'projection'),
#             image, proj_data,
#             projection_model=pm)
#         back_proj = BackProjectionToR(self.info.child_scope(
#             'backprojection'), image, proj, projection_model=pm)
#         result = image / inputs[KT.EFFICIENCY_MAP] * back_proj
#         return result
