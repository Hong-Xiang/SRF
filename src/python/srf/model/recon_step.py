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
            submodels=subgraphs,
            config=config)

    @classmethod
    def maker_builder(self):
        def builder(inputs):
            def maker(graph, n):
                return ReconStep(graph.info.child_scope(n), inputs=inputs)
            return maker
        return builder

    def kernel(self, inputs):
        KT, KS = self.KEYS.TENSOR, self.KEYS.SUBGRAPH
        image, proj_data = inputs[KT.IMAGE], inputs[KT.PROJECTION_DATA]
        proj = self.subgraph(
            KS.PROJECTION, SubgraphMakerFinder(image, proj_data))()
        back_proj = self.subgraph(
            KS.BACKPROJECTION, SubgraphMakerFinder(proj, image))
        result = image / inputs[KT.EFFICIENCY_MAP] * back_proj
        return result


class ReconStepHardCoded(ReconStep):
    def __init__(self, info, *, inputs, config=None):
        super().__init__(info, inputs=inputs, config=config)

    def kernel(self, inputs):
        KT, KS = self.KEYS.TENSOR, self.KEYS.SUBGRAPH
        image, proj_data = inputs[KT.IMAGE], inputs[KT.PROJECTION_DATA]
        from ..physics import ToRModel
        from .projection import ProjectionToR
        from .backprojection import BackProjectionToR
        pm = ToRModel('projection_model')
        proj = ProjectionToR(self.info.child_scope(
            'projection'),
            image, proj_data,
            projection_model=pm)
        back_proj = BackProjectionToR(self.info.child_scope(
            'backprojection'), image, proj, projection_model=pm)
        result = image / inputs[KT.EFFICIENCY_MAP] * back_proj
        return result
