from dxl.data.tensor import Tensor


class Image(Tensor):
    def __init__(self, data, center, size):
        super().__init__(data)
        self.center = center
        self.size = size

    @property
    def grid(self):
        return self.shape
