from dxl.learn.core import Model


class ReconStep(Model):
    class KEYS(Model.KEYS):
        class TENSOR(Model.KEYS.TENSOR):
            IMAGE = 'image'
            EFFICIENCY_MAP = 'efficiency_map'
            PROJECTION_DATA = 'projection_data'

        class SUBGRAPH(Model.KEYS.SUBGRAPH):
            PROJECTION = 'projection'
            BACK_PROJECTION = 'back_projection'

    def __init__(self, info,
                 image,
                 projection_data,
                 efficiency_map,
                 *,
                 subgraphs=None,
                 config=None,
                 ):
        super().__init__(
            name,
            {
                self.KEYS.TENSOR.IMAGE:
                image,
                self.KEYS.TENSOR.EFFICIENCY_MAP:
                efficiency_map,
                self.KEYS.TENSOR.PROJECTION_DATA:
                projection_data
            },
            graph_info=graph_info)

    def kernel(self, inputs):
        KT = self.KEYS.TENSOR
        image = self.tensor(KT.IMAGE)
        proj_data = self.subgraph()
        proj = image.projection(projection_model, proj_data)
        back_proj = proj.back_projection(projection_model, image)
        result = image / self.tensor(KT.EFFICIENCY_MAP) * back_proj
        return Tensor(result, self.info.child_tensor('result'))
