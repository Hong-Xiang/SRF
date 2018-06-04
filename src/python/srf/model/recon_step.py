from dxl.learn.core import Model
from dxl.learn.core import SubgraphMakerFinder


class ReconStep(Model):
    class KEYS(Model.KEYS):
        class TENSOR(Model.KEYS.TENSOR):
            IMAGE = 'image'
            EFFICIENCY_MAP = 'efficiency_map'
            PROJECTION_DATA = 'projection_data'

        class SUBGRAPH(Model.KEYS.SUBGRAPH):
            PROJECTION = 'projection'
            BACKPROJECTION = 'backprojection'

    def __init__(self, info,
                 *,
                 inputs,
                 subgraphs=None,
                 config=None,):
        super().__init__(
            info,
            inputs=inputs,
            subgraphs=subgraphs,
            config=config)

    def kernel(self, inputs):
        KT, KS = self.KEYS.TENSOR, self.KEYS.SUBGRAPH
        image, proj_data = inputs[KT.IMAGE], inputs[KT.PROJECTION_DATA]
        proj = self.subgraph(
            KS.PROJECTION, SubgraphMakerFinder(image, proj_data))()
        back_proj = self.subgraph(
            KS.BACKPROJECTION, SubgraphMakerFinder(proj, image))
        result = image / inputs[KT.EFFICIENCY_MAP] * back_proj
        return result
