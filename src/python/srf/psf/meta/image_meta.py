import attr
import numpy as np

from srf2.core.abstracts import Meta, Singleton

__all__ = (
    'Image_meta', 'Image_meta_2d', 'Image_meta_3d',
    'Image_meta_singleton', 'Image_meta_2d_singleton', 'Image_meta_3d_singleton',)


@attr.s
class Image_meta(Meta):
    _shape = attr.ib(default = (1,) * 3)
    _center = attr.ib(default = (0,) * 3)
    _size = attr.ib(default = (1,) * 3)
    _dims = attr.ib(default = ('x', 'y', 'z'))

    @_shape.validator
    def _check_arg_shape(self, attribute, value):
        if not 2 <= len(value) <= 4:
            raise ValueError('_shape must in range [2, 4]')
        if len(value) == 4:
            raise NotImplementedError
        if not isinstance(value, tuple):
            raise TypeError('_shape must be a tuple')

    @_center.validator
    def _check_arg_center(self, attribute, value):
        if len(value) != len(self._shape):
            raise ValueError('_center must have same size with _shape')
        if not isinstance(value, tuple):
            raise TypeError('_center must be a tuple')

    @_size.validator
    def _check_arg_size(self, attribute, value):
        if len(value) != len(self._shape):
            raise ValueError('_size must have same size with _shape')
        if not isinstance(value, tuple):
            raise TypeError('_size must be a tuple')

    @_dims.validator
    def _check_arg_dims(self, attribute, value):
        if len(value) != len(self._shape):
            raise ValueError('_dims must have same size with _shape')
        if not isinstance(value, tuple):
            raise TypeError('_dims must be a tuple')
        if not set(value).issubset({'x', 'y', 'z', 't'}):
            return ValueError('Only x, y, z and t are allowed int _dims')
        if 't' in value:
            raise NotImplementedError

    @property
    def shape(self):
        return self._shape

    @property
    def center(self):
        return self._center

    @property
    def size(self):
        return self._size

    @property
    def dims(self):
        return self._dims

    @property
    def unit_size(self):
        return tuple(ti / tj for (ti, tj) in zip(self._size, self._shape))

    @property
    def ndim(self):
        return len(self._dims)

    @property
    def n_x(self):
        return self._shape[self._dims.index('x')] if 'x' in self._dims else 1

    @property
    def n_y(self):
        return self._shape[self._dims.index('y')] if 'y' in self._dims else 1

    @property
    def n_xy(self):
        return self.n_x * self.n_y

    @property
    def n_z(self):
        return self._shape[self._dims.index('z')] if 'z' in self._dims else 1

    @property
    def n_t(self):
        return self._shape[self._dims.index('t')] if 't' in self._dims else 1

    @property
    def n_all(self):
        return np.array(self.shape).prod()

    def map(self, f):
        # raise NotImplementedError
        return Image_meta(*f(tuple(self.shape), tuple(self.center), tuple(self.size),
                             tuple(self.dims)))

    def meshgrid(self):
        raise NotImplementedError

    def __getitem__(self, ind):
        def slice_kernel(ind):
            def kernel(shape, center, size, dims):
                shape = list(shape)
                center = list(center)
                size = list(size)

                for k in range(len(shape)):
                    start, stop, step = ind[k].start, ind[k].stop, ind[k].step

                    if step is None:
                        step = 1

                    rang = range(shape[k])[start: stop: step]
                    unit_size = size[k] / shape[k]
                    center[k] = (rang[0] + rang[-1] + 1) / 2 * unit_size + center[k] - size[k] / 2
                    shape[k] = len(rang)
                    size[k] = shape[k] * unit_size * step
                return tuple(shape), tuple(center), tuple(size), dims

            return kernel

        return self.map(slice_kernel(ind))

    def locate(self, pos = None):
        if pos is None:
            pos = (0,) * self.ndim
        result = [0] * self.ndim
        for k in range(self.ndim):
            result[k] = (pos[k] - self.center[k] + self.size[k] / 2) / self.unit_size[k] - 0.5
        return tuple(result)

    def transpose(self, perm = None):
        raise NotImplementedError


class Image_meta_2d(Image_meta):
    def __init__(self, shape = (1, 1), center = (0, 0), size = (1, 1), dims = ('x', 'y')):
        if len(shape) != 2:
            raise ValueError(self.__class__, ' is only consistent with 2D case')
        super().__init__(shape, center, size, dims)

    def map(self, f):
        return Image_meta_2d(
            *f(tuple(self.shape), tuple(self.center), tuple(self.size), tuple(self.dims)))

    # def meshgrid(self, slice = None):
    #     if slice is None:
    #         x = np.arange(self.shape[0]) * self.unit_size[0] + self.center[0] - self.size[0] / 2 + \
    #             self.unit_size[0] / 2
    #         y = np.arange(self.shape[1]) * self.unit_size[1] + self.center[1] - self.size[1] / 2 + \
    #             self.unit_size[1] / 2
    #     else:
    #         x = np.arange(self.shape[0])[slice[0]] * self.unit_size[0] + self.center[0] - self.size[
    #             0] / 2 + self.unit_size[0] / 2
    #         y = np.arange(self.shape[1])[slice[1]] * self.unit_size[1] + self.center[1] - self.size[
    #             1] / 2 + self.unit_size[1] / 2
    #
    #     y1, x1 = np.meshgrid(y, x)
    #     return x1, y1
    def meshgrid(self, slice = None):
        if slice is None:
            x = np.arange(self.shape[0])
            y = np.arange(self.shape[1])
        else:
            x = np.arange(self.shape[0])[slice[0]]
            y = np.arange(self.shape[1])[slice[1]]

        y1, x1 = np.meshgrid(y, x)
        return x1, y1

    def grid_centers(self, slice = None):
        x1, y1 = self.meshgrid(slice)
        pos_x = x1 * self.unit_size[0] + self.center[0] - self.size[0] / 2 + self.unit_size[0] / 2
        pos_y = y1 * self.unit_size[1] + self.center[1] - self.size[1] / 2 + self.unit_size[1] / 2
        return pos_x, pos_y

    def theta(self):
        x1, y1 = self.grid_centers()
        return np.arctan2(y1, x1)

    def polar_meshgrid(self):
        x1, y1 = self.meshgrid()
        theta = self.theta()
        radius = np.sqrt(x1 ** 2 + y1 ** 2)
        return radius, theta

    def transpose(self, perm = None):
        if not perm:
            perm = range(self.ndim)[::-1]
        if 't' in perm:
            raise NotImplementedError

        if set(perm).issubset({'x', 'y', 'z'}):
            perm = [self.dims.index(e) for e in perm]

        shape = tuple([self.shape[i] for i in perm])
        center = tuple([self.center[i] for i in perm])
        size = tuple([self.size[i] for i in perm])
        dims = tuple([self.dims[i] for i in perm])
        return Image_meta_2d(shape, center, size, dims)


class Image_meta_3d(Image_meta):
    def __init__(self, shape = (1, 1, 1), center = (0, 0, 0), size = (1, 1, 1),
                 dims = ('x', 'y', 'z')):
        if len(shape) != 3:
            raise ValueError(self.__class__, ' is only consistent with 2D case')
        super().__init__(shape, center, size, dims)

    def map(self, f):
        return Image_meta_3d(
            *f(tuple(self.shape), tuple(self.center), tuple(self.size), tuple(self.dims)))

    # def meshgrid(self, slice = None):
    #     if slice is None:
    #         x = np.arange(self.shape[0]) * self.unit_size[0] + self.center[0] - self.size[0] / 2 + \
    #             self.unit_size[0] / 2
    #         y = np.arange(self.shape[1]) * self.unit_size[1] + self.center[1] - self.size[1] / 2 + \
    #             self.unit_size[1] / 2
    #         z = np.arange(self.shape[2]) * self.unit_size[2] + self.center[2] - self.size[2] / 2 + \
    #             self.unit_size[2] / 2
    #     else:
    #         x = np.arange(self.shape[0])[slice[0]] * self.unit_size[0] + self.center[0] - self.size[
    #             0] / 2 + self.unit_size[0] / 2
    #         y = np.arange(self.shape[1])[slice[1]] * self.unit_size[1] + self.center[1] - self.size[
    #             1] / 2 + self.unit_size[1] / 2
    #         z = np.arange(self.shape[2])[slice[2]] * self.unit_size[2] + self.center[2] - self.size[
    #             2] / 2 + self.unit_size[2] / 2
    #     (y1, x1, z1) = np.meshgrid(y, x, z)
    #     return x1, y1, z1
    #
    # def meshgrid_2d(self, slice = None):
    #     if slice is None:
    #         x = np.arange(self.shape[0]) * self.unit_size[0] + self.center[0] - self.size[0] / 2 + \
    #             self.unit_size[0] / 2
    #         y = np.arange(self.shape[1]) * self.unit_size[1] + self.center[1] - self.size[1] / 2 + \
    #             self.unit_size[1] / 2
    #     else:
    #         x = np.arange(self.shape[0])[slice[0]] * self.unit_size[0] + self.center[0] - self.size[
    #             0] / 2 + self.unit_size[0] / 2
    #         y = np.arange(self.shape[1])[slice[1]] * self.unit_size[1] + self.center[1] - self.size[
    #             1] / 2 + self.unit_size[1] / 2
    #     (y1, x1) = np.meshgrid(y, x)
    #     return x1, y1
    #
    # def meshgrid_1d(self, slice = None):
    #     if slice is None:
    #         z = np.arange(self.shape[2]) * self.unit_size[2] + self.center[2] - self.size[2] / 2 + \
    #             self.unit_size[2] / 2
    #     else:
    #         z = np.arange(self.shape[2])[slice[2]] * self.unit_size[2] + self.center[2] - self.size[
    #             2] / 2 + self.unit_size[2] / 2
    #     return z

    def meshgrid(self, slice = None):
        if slice is None:
            x = np.arange(self.shape[0])
            y = np.arange(self.shape[1])
            z = np.arange(self.shape[2])
        else:
            x = np.arange(self.shape[0])[slice[0]]
            y = np.arange(self.shape[1])[slice[1]]
            z = np.arange(self.shape[2])[slice[2]]
        (y1, x1, z1) = np.meshgrid(y, x, z)
        return x1, y1, z1

    def grid_centers(self, slice = None):
        x1, y1, z1 = self.meshgrid(slice)
        pos_x = x1 * self.unit_size[0] + self.center[0] - self.size[0] / 2 + self.unit_size[0] / 2
        pos_y = y1 * self.unit_size[1] + self.center[1] - self.size[1] / 2 + self.unit_size[1] / 2
        pos_z = z1 * self.unit_size[2] + self.center[2] - self.size[2] / 2 + self.unit_size[2] / 2
        return pos_x, pos_y, pos_z

    def meshgrid_2d(self, slice = None):
        if slice is None:
            x = np.arange(self.shape[0])
            y = np.arange(self.shape[1])
        else:
            x = np.arange(self.shape[0])[slice[0]]
            y = np.arange(self.shape[1])[slice[1]]
        (y1, x1) = np.meshgrid(y, x)
        return x1, y1

    def grid_centers_2d(self, slice = None):
        x1, y1 = self.meshgrid_2d(slice)
        pos_x = x1 * self.unit_size[0] + self.center[0] - self.size[0] / 2 + self.unit_size[0] / 2
        pos_y = y1 * self.unit_size[1] + self.center[1] - self.size[1] / 2 + self.unit_size[1] / 2
        return pos_x, pos_y

    def meshgrid_1d(self, slice = None):
        if slice is None:
            z = np.arange(self.shape[2])
        else:
            z = np.arange(self.shape[2])[slice[2]]
        return z

    def grid_centers_1d(self, slice = None):
        z1 = self.meshgrid_1d(slice)
        pos_z = z1 * self.unit_size[2] + self.center[2] - self.size[2] / 2 + self.unit_size[2] / 2
        return pos_z

    def theta(self):
        x1, y1 = self.grid_centers_2d()
        return np.arctan2(y1, x1)

    def cylindral_meshgrid(self):
        x1, y1, z1 = self.meshgrid()
        theta = self.theta()
        radius = np.sqrt(x1 ** 2 + y1 ** 2)
        return radius, theta, z1

    def transpose(self, perm = None):
        if not perm:
            perm = range(self.ndim)[::-1]
        if 't' in perm:
            raise NotImplementedError

        if set(perm).issubset({'x', 'y', 'z'}):
            perm = [self.dims.index(e) for e in perm]

        shape = tuple([self.shape[i] for i in perm])
        center = tuple([self.center[i] for i in perm])
        size = tuple([self.size[i] for i in perm])
        dims = tuple([self.dims[i] for i in perm])
        return Image_meta_3d(shape, center, size, dims)


class Image_meta_singleton(Image_meta, Singleton):
    pass


class Image_meta_2d_singleton(Image_meta_2d, Singleton):
    pass


class Image_meta_3d_singleton(Image_meta_3d, Singleton):
    pass
