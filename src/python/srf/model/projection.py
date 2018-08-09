from dxl.learn.core import Model, Constant, Tensor
# from ..physics import ToRModel
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


# class ProjectionToR(Projection):
#     class KEYS(Projection.KEYS):
#         class GRAPH(Projection.KEYS.GRAPH):
#             SPLIT = 'split'
#     AXIS = ('x', 'y', 'z')

#     def __init__(self,
#                  projection_model=None,
#                  info=None,
#                  ):
#         info = info or 'projection_tor'
#         super().__init__(info)
#         if projection_model is None:
#             projection_model = ToRModel('projection_model')
#         self.projection_model = projection_model

#     def kernel(self, inputs):
#         KT = self.KEYS.TENSOR
#         proj_data, image = inputs[KT.PROJECTION_DATA], inputs[KT.IMAGE]
#         self.projection_model.check_inputs(
#             proj_data, self.KEYS.TENSOR.PROJECTION_DATA)

#         results = {}
#         pm = self.projection_model
#         for a in pm.AXIS:
#             results[a] = pm.projection(
#                 lors=proj_data[a], image=image.transpose(pm.perm(a)))
#         return results


class ProjectionOrdinary(Projection):
    """
    A unified projection entry.
    """
    def __init__(self,
                 physical_model,
                 info=None):
        info = info or 'projection_ordinary'
        super().__init__(info)
        self.physical_model = physical_model

    def kernel(self, inputs):
        KT = self.KEYS.TENSOR
        self.physical_model.check_inputs(
            inputs[KT.PROJECTION_DATA], KT.PROJECTION_DATA)
        pm = self.physical_model
        result = pm.projection(lors = inputs[KT.PROJECTION_DATA], image=inputs[KT.IMAGE])
        return result