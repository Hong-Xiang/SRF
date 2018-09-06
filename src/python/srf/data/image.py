from dxl.learn.tensor import shape, transpose, initializer, assign


class Image:
    """
    Image data with center and size info.
    """

    def __init__(self, data, center, size):
        self.data = data
        self.center = center
        self.size = size

    @property
    def grid(self):
        return shape(self.data)

    def fmap(self, f):
        return Image(f(self.data), self.center, self.size)


@transpose.register(Image)
def _(t, perm=None):
    if perm is None:
        perm = [2, 1, 0]
    center = [t.center[p] for p in perm]
    size = [t.size[p] for p in perm]
    return Image(transpose(t.data, perm), center=center, size=size)


@shape.register(Image)
def _(t):
    return shape(t.data)


@initializer.register(Image)
def _(t):
    return initializer(t.data)


@assign.register(Image)
def _(img, target):
    return Image(img.assign(target), img.center, img.size)
