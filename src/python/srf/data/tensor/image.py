from dxl.learn.core import Tensor, Variable


class Image:
    def __init__(self, data, center, size):
        self.data = data
        self.center = center
        self.size = size

    def transpose(self, perm=None):
        if perm is None:
            perm = [2, 1, 0]
        image = transpose(perm)
        center = [self.center[p] for p in perm]
        size = [self.size[p] for p in perm]
        return Image(image, center=center, size=size)
