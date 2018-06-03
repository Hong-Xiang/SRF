from dxl.learn import Tensor, Variable


class Image(Tensor):
    def __init__(self, grid, center, size):
        pass

    def split_xyz(self, graph):
        pass


class ImageVariable(Variable):
    def as_tensor(self):
        # No new tensorflow tensor is created, just for unified high order interface.
        return Image()


class ImageXYZ(Tensor):
    def __init__(self, image: Image, is_roate):
        pass

    def rotation(self):
        pass

    def merge(self):
        return Image()
