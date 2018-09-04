from doufo.tensor import shape, transpose


class Image:
    def __init__(self, data, center, size):
        self.data = data
        self.center = center
        self.size = size

    @property
    def grid(self):
        return shape(self.data)

    def transpose(self, perm=None):
        if perm is None:
            perm = [2, 1, 0]
        center = [self.center[p] for p in perm]
        size = [self.size[p] for p in perm]
        return Image(transpose(self.data, perm), center=center, size=size)
