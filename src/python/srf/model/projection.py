from dxl.learn.core import Model, Constant, Tensor
from ..physics import ToRModel

from ..tensor import Image


class Projection(Model):
    class KEYS(Model.KEYS):
        class TENSOR(Model.KEYS.TENSOR):
            IMAGE = 'image'
            PROJECTION_DATA = 'projection_data'

    def __init__(self, info):
        """
        Args:
        physics:
        """
        super().__init__(info)

    def kernel(self, inputs):
        raise NotImplementedError


class ProjectionToR(Projection):
    class KEYS(Projection.KEYS):
        class GRAPH(Projection.KEYS.GRAPH):
            SPLIT = 'split'
    AXIS = ('x', 'y', 'z')

    def __init__(self,
                 model=None,
                 info=None,
                 ):
        info = info or 'projection_tor'
        super().__init__(info)
        if model is None:
            model = ToRModel('projection_model')
        self.model = model

    def kernel(self, inputs):
        KT = self.KEYS.TENSOR
        proj_data, image = inputs[KT.PROJECTION_DATA], inputs[KT.IMAGE]
        # TODO: Add ProjectionModelLocator
        self.model.check_inputs(
            proj_data, self.KEYS.TENSOR.PROJECTION_DATA)
        imgz = image.transpose()
        imgx = image.transpose(perm=[2, 0, 1])
        imgy = image.transpose(perm=[1, 0, 2])
        imgs = {'x': imgx, 'y': imgy, 'z': imgz}
        results = {}
        for a in self.AXIS:
            results[a] = self.model.projection(
                lors=proj_data[a], image=imgs[a],)
        return results
