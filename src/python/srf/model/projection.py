from dxl.learn.core import Model, Constant, Tensor
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


class ProjectionToR(Projection):
    class KEYS(Projection.KEYS):
        class SUBGRAPH(Projection.KEYS.SUBGRAPH):
            SPLIT = 'split'
    AXIS = ('x', 'y', 'z')

    def __init__(self,
                 info,
                 image,
                 projection_data,
                 *,
                 projection_model=None,
                 config=None,
                 ):
        self.projection_model = projection_model
        super().__init__(
            info,
            image=image, projection_data=projection_data, config=config
        )

    def kernel(self, inputs):
        KT = self.KEYS.TENSOR
        proj_data, image = inputs[KT.PROJECTION_DATA], inputs[KT.IMAGE]
        # TODO: Add ProjectionModelLocator
        if self.projection_model is None:
            self.projection_model = ToRModel('projection_model')
        self.projection_model.check_inputs(
            proj_data, self.KEYS.TENSOR.PROJECTION_DATA)
        imgz = image.transpose()
        imgx = image.transpose(perm=[2, 0, 1])
        imgy = image.transpose(perm=[1, 0, 2])
        imgs = {'x': imgx, 'y': imgy, 'z': imgz}
        results = {}
        for a in self.AXIS:
            results[a] = self.projection_model.projection(lors=proj_data[a],
                                                          image=imgs[a],)
        return results
