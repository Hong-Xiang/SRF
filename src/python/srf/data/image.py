from dxl.learn.core import Tensor

# TODO: Implement Unified Tensor among doufo and dxl.learn.core

# class Image(Tensor):
#     def __init__(self, data, center, size):
#         super().__init__(data)
#         self.center = center
#         self.size = size

#     @property
#     def grid(self):
#         return self.shape

class Image(Tensor):
    def __init__(self, data, center, size):
        super().__init__(data)
        self.grid = data.shape
        self.center = center
        self.size = size

    def transpose(self, perm=None):
        if perm is None:
            perm = [2, 1, 0]
        image = super().transpose(perm)
        grid = [self.grid[p] for p in perm]
        center = [self.center[p] for p in perm]
        size = [self.size[p] for p in perm]
        return Image(image, center=center, size=size)
