from ..physics import ToRModel
from dxl.learn.model import Summation
from dxl.learn.core import Model

from srf.physics import ToRModel


class BackProjection(Model):
    class KEYS(Model.KEYS):
        class TENSOR(Model.KEYS.TENSOR):
            IMAGE = 'image'
            PROJECTION_DATA = 'projection_data'

    def __init__(self, info=None):
        info = info or 'backprojection'
        super().__init__(info)

    def kernel(self, inputs):
        raise NotImplementedError


class BackProjectionToR(BackProjection):
    class KEYS(BackProjection.KEYS):
        class GRAPH(BackProjection.KEYS.GRAPH):
            SPLIT = 'split'

    def __init__(self, projection_model=None, info=None):
        super().__init__(info)
        if projection_model is None:
            projection_model = ToRModel('projection_model')
        self.projection_model = projection_model

    def kernel(self, inputs):
        KT = self.KEYS.TENSOR
        image, lors = inputs[KT.IMAGE], inputs[KT.PROJECTION_DATA]
        lors = {
            a: {
                'lors': lors['lors'][a],
                'lors_value': lors['lors_value'][a],
            }
            for a in self.projection_model.AXIS
        }
        result = {}
        pm = self.projection_model
        for a in self.projection_model.AXIS:
            result[a] = pm.backprojection(lors[a], image)
            result[a] = result[a].transpose(pm.perm('z'))
        result = {'z': result['z']}
        result = Summation(self.info.name / 'summation')(result)
        return result
