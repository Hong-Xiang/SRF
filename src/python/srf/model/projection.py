from dxl.learn.core import Model, Constant
from ..physics import ToRModel

from ..tensor import Image


class Projection(Model):
    class KEYS(Model.KEYS):
        class TENSOR(Model.KEYS.TENSOR):
            IMAGE = 'image'
            PROJECTION_DATA = 'projection_data'

    def __init__(self, info, image, projection_data, *, config=None):
        """
        Args:
        physics:
        """
        super().__init__(info, inputs={
            self.KEYS.TENSOR.IMAGE: image,
            self.KEYS.TENSOR.PROJECTION_DATA: projection_data
        }, config=config)

    def kernel(self, inputs):
        raise NotImplementedError


class ProjectionTOR(Projection):
    class KEYS(Projection.KEYS):
        class SUBGRAPH(Projection.KEYS.SUBGRAPH):
            SPLIT = 'split'
    AXIS = ('x', 'y', 'z')

    def __init__(self,
                 info,
                 image,
                 projection_data
                 * ,
                 config
                 ):
        self.projection_model = ToRModel(self.info.name / 'projection_model')
        self.projection_model.check_inputs(
            projection_data, self.KEYS.TENSOR.PROJECTION_DATA)
        super().__init__(
            info,
            {self.KEYS.TENSOR.IMAGE: image,
                self.KEYS.TENSOR.PROJECTION_DATA: projection_data},
        )

    def kernel(self, inputs):
        KT = self.KEYS.TENSOR
        imgz = image.transpose()
        imgx = image.transpose(perm=[2, 0, 1])
        imgy = image.transpose(perm=[1, 0, 2])
        imgs = {'x': imgx, 'y': imgy, 'z': imgz}
        proj_data = inputs[KT.PROJECTION_DATA]
        results = {}
        pm = self.get_or_create_projection_model()
        for a in self.AXIS:
            results[a] = pm().projection(lors=proj_data[a],
                                         image=imgs[a],)
        return results
