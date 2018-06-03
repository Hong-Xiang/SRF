from dxl.learn import Tensor, Variable


class Image(Tensor):
    def __init__(self, data, center, size, info=None):
        super().__init__(data, info)
        self.grid = data.shape
        self.center = center
        self.size = size

    def transpose(self, perm=None):
        if perm is None:
            perm = [2, 1, 0]
        image = self.transpose(perm)
        grid = [self.grid[p] for p in perm]
        center = [self.center[p] for p in perm]
        size = [self.size[p] for p in perm]
        return Image(image, gird=grid, center=center, size=size)
