from ..physics import ToRModel
from dxl.learn.model import Summation
from dxl.learn.core import Model

from srf.physics import ToRModel


class BackProjection(Model):
    class KEYS(Model.KEYS):
        class TENSOR(Model.KEYS.TENSOR):
            IMAGE = 'image'
            PROJECTION_DATA = 'projection_data'

    def __init__(self, info, image, projection_data, *, config):
        self._projection_model = None
        super().__init__(info, inputs={
            self.KEYS.TENSOR.IMAGE: image,
            self.KEYS.TENSOR.PROJECTION_DATA: projection_data
        }, config=config)

    def kernel(self, inputs):
        raise NotImplementedError


class BackProjectionToR(BackProjection):
    class KEYS(BackProjection.KEYS):
        class GRAPH(BackProjection.KEYS.GRAPH):
            SPLIT = 'split'

    def __init__(self, info,
                 projection_data,
                 image,
                 *,
                 projection_model=None,
                 config=None):
        self.projection_model = projection_model
        super().__init__(
            info,
            image,
            projection_data,
            config=config)

    def kernel(self, inputs):
        KT = self.KEYS.TENSOR
        image, lors = inputs[KT.IMAGE], inputs[KT.PROJECTION_DATA]
        if self.projection_model is None:
            self.projection_model = ToRModel('projection_model')
        # self.projection_model.check_inputs(
            # lors, self.KEYS.TENSOR.PROJECTION_DATA)
        # self.projection_model.check_inputs(imgs, self.KEYS.TENSOR.IMAGE)
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
