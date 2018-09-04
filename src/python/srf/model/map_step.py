from dxl.learn.core import Model
from srf.tensor import Image


class MapStep(Model):
    """ A general model to compute the efficiency map.

    A MapStep describes the efficiency map computing procedure.
    The main step is backprojection. Input lors are backprojected to the 
    image and form the efficiency map.
    The physical model is specified by input so that this class can be 
    commonly used by different physical models such as ToR or Siddon.

    """

    class KEYS(Model.KEYS):
        class TENSOR(Model.KEYS.TENSOR):
            IMAGE = 'image'
            LORS = 'lors'
            LORS_VALUE = 'lors_value'

        class GRAPH(Model.KEYS.GRAPH):
            BACKPROJECTION = 'backprojection'

    def __init__(self, name, backprojection):
        super().__init__(name)
        self.backprojection = backprojection

    def kernel(self, image, lors, lors_value):
        return self.backprojection(lors, lors_value, image)
