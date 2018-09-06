from dxl.learn import Model
from doufo import multidispatch

__all__ = ['ProjectionOrdinary', 'projection']


class Projection(Model):
    def __init__(self, name, physics_model):
        super().__init__(name)
        self.physics_model = physics_model

    def kernel(self, image, projection_data):
        raise NotImplementedError


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

    def kernel(self, image, projection_data):
        return projection(self.physical_model, image, projection_data)

    @property
    def parameters(self):
        return []


@multidispatch(nargs=3, nouts=1)
def projection(physics, image, projection_data):
    raise NotImplementedError(
        f"No projection implementation for physics model: {type(physics)}; image: {type(image)}; projection_data: {type(projection_data)}.")
