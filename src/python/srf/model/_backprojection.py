from dxl.learn import Model
from doufo import multidispatch

__all__ = ['BackProjectionOrdinary', 'backprojection', 'MapOrdinary', 'map_lors']


class BackProjection(Model):
    def __init__(self, name='backprojection'):
        super().__init__(name)

    def build(self, *args):
        pass

    def kernel(self, projection_data, image):
        raise NotImplementedError


class BackProjectionOrdinary(BackProjection):
    """
    A unified backprojection entry.
    """

    def __init__(self,
                 physical_model,
                 name='backprojection_ordinary'):
        super().__init__(name)
        self.physical_model = physical_model

    def kernel(self, projection_data, image):
        return backprojection(self.physical_model, projection_data, image)

    @property
    def parameters(self):
        return []


@multidispatch(nargs=3, nouts=1)
def backprojection(physical_model, projection_data, image):
    raise NotImplementedError


class MapOrdinary(BackProjection):
    """
    A unified backprojection entry.
    """

    def __init__(self,
                 physical_model,
                 name=None):
        super().__init__(name)
        self.physical_model = physical_model

    def kernel(self, projection_data, image):
        return map_lors(self.physical_model, projection_data, image)


@multidispatch(nargs=3, nouts=1)
def map_lors(physical_model, projection_data, image):
    return physical_model.map_lors(projection_data, image)
