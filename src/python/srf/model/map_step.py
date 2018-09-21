from srf.data import Image
from dxl.learn import Model


class MapStep(Model):
    """ A general model to compute the efficiency map.

    A MapStep describes the efficiency map computing procedure.
    The main step is backprojection. Input lors are backprojected to the 
    image and form the efficiency map.
    The physical model is specified by input so that this class can be 
    commonly used by different physical models such as ToR or Siddon.

    """
    class KEYS(Model.KEYS):
        class TENSOR:
            IMAGE = 'image'
            PROJECTION_DATA = 'projection_data'

    def __init__(self, name, backprojection):
        super().__init__(name)
        self.backprojection = backprojection

    # def kernel(self, image: Image, projection_data):
    #     return self.backprojection(projection_data, image)
    def build(self, *args):
        pass
    def kernel(self,inputs):
        image = inputs[self.KEYS.TENSOR.IMAGE]
        projection_data = inputs[self.KEYS.TENSOR.PROJECTION_DATA]
        return self.backprojection(projection_data,image)

    @property
    def parameters(self):
        return []