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

    def __init__(self, info,
                 *,
                 backprojection=None,
                 graphs=None,
                 config=None):
        if graphs is None:
            graphs = {}
        if backprojection is not None:
            graphs.update(
                {self.KEYS.GRAPH.BACKPROJECTION: backprojection}
            )
        super().__init__(
            info,
            graphs=graphs,
            config=config)

    def kernel(self, inputs):
        KT, KS = self.KEYS.TENSOR, self.KEYS.GRAPH
        image, lors, lors_value = inputs[KT.IMAGE], inputs[KT.LORS], inputs[KT.LORS_VALUE]
        effmap = self.graphs[KS.BACKPROJECTION]({
            'projection_data':{'lors':lors, 'lors_value': lors_value},
            'image':image
        })
        return effmap
