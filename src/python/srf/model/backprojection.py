from ..physics import ToRModel
from dxl.learn.model import Summation


class BackProjection(object):
    class KEYS(Model.KEYS):
        class TENSOR(Model.KEYS.TENSOR):
            IMAGE = 'image'
            PROJECTION_DATA = 'projection_data'

    def __init__(self, image, projection_data, *, config):
        self._projection_model = None
        super().__init__(info, inputs={
            self.KEYS.TENSOR.IMAGE: image,
            self.KEYS.TENSOR.PROJECTION_DATA: projection_data
        }, config=config)

    def kernel(self, inputs):
        raise NotImplementedError


class BackProjectionToR(BackProjection):
    class KEYS(BackProjection.KEYS):
        class SUBGRAPH(BackProjection.KEYS.SUBGRAPH):
            SPLIT = 'split'

    def __init__(self, info,
                 projection_data,
                 image,
                 *,
                 projection_model=None,
                 config=None):
        if projection_model is None:
            projection_model = TORModel(self.info.name / 'projection_model')
        self.projection_model = projection_model
        self.projection_model.check_inputs(
            projection_data, self.KEYS.TENSOR.PROJECTION_DATA)
        self.projection_model.check_inputs(image, self.KEYS.TENSOR.IMAGE)
        super().__init__(
            info,
            {self.KEYS.TENSOR.IMAGE: image,
                self.KEYS.TENSOR.LORS: lors}, config=config)

    def kernel(self, inputs):
        KT = self.KEYS.TENSOR
        imgs = inputs[KT.IMAGE]
        lors = inputs[KT.PROJECTION_DATA]
        result = {}
        pm = self.projection_model
        for a in self.AXIS:
            lors[a] = lors[a].transpose()
            result[a] = pm.backprojection(lors[a], imgs[a])
            result[a] = result[a].transpose(pm.perm(a))
        result = Summation(self.info.name / 'summation', result)
        return result
