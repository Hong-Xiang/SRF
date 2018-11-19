import h5py
import numpy as np

from srf2.meta.image_meta import *

__all__ = ('Image', 'Image_2d', 'Image_3d',)


# noinspection PyPep8Naming
class Image:
    def __init__(self, data = None, meta: Image_meta = None):
        if meta is None and data is None:
            meta = Image_meta()
            data = np.zeros(meta.shape, dtype = np.float32)

        if data is None:
            data = np.zeros(meta.shape, dtype = np.float32)

        if meta is None:
            meta = Image_meta(data.shape)

        if not isinstance(data, np.ndarray):
            raise TypeError('data must be a np.ndarray object')

        if data.shape != meta.shape:
            raise ValueError('data must has same shape with its meta')

        data.astype(np.float32)

        self._data = data
        self._meta = meta

    def __eq__(self, other):
        if self.__class__ != other.__class__:
            return False

        return self.meta == other.meta and np.array_equal(self.data, other.data)

    @property
    def data(self):
        return self._data

    @property
    def meta(self):
        return self._meta

    @property
    def T(self):
        return self.transpose()

    def normalize(self):
        sum_data = np.sum(self.data)
        return Image(self.data / sum_data, self.meta)

        # return Image(self.data / sum_data / self.meta.unit_size[0] / self.meta.unit_size[1] /
        #              self.meta.unit_size[2], self.meta)

    def transpose(self, perm = None):
        if perm is None:
            perm = np.arange(self.meta.ndim)[::-1]

        if set(perm).issubset({'x', 'y', 'z'}):
            perm = [self.meta.dims.index(e) for e in perm]

        return Image(self.data.transpose(perm), self.meta.transpose(perm))

    def map(self, f):
        return Image(*f(self.data, self.meta))

    def __getitem__(self, item):
        def slice_kernel(item):
            def kernel(data, meta):
                return data[item], meta[item]

            return kernel

        return self.map(slice_kernel(item))

    def save_h5(self, path = None, mode = 'w'):
        if path is None:
            path = 'tmp' + self.__class__.__name__ + '.h5'
        self.meta.save_h5(path, mode)
        with h5py.File(path, 'r+') as fout:
            group = fout.create_group('Image_data')
            group.create_dataset('_data', data = self.data, compression = "gzip")

    @classmethod
    def load_h5(cls, path = None):
        if path is None:
            path = 'tmp' + cls.__name__ + '.h5'
        meta = Image_meta.load_h5(path)
        with h5py.File(path, 'r') as fin:
            dataset = fin['Image_data']
            data = np.array(dataset['_data'])
            return cls(data, meta)


# noinspection PyPep8Naming
class Image_2d(Image):
    def __init__(self, data = None, meta: Image_meta_2d = None):
        super().__init__(data, meta)
        if meta.ndim != 2:
            raise ValueError(self.__class__.__name__, ' is only consistent with 2D case')

    def map(self, f):
        return Image_2d(*f(self.data, self.meta))


# noinspection PyPep8Naming
class Image_3d(Image):
    def __init__(self, data = None, meta: Image_meta_3d = None):
        if meta is None and data is None:
            meta = Image_meta_3d()
            data = np.zeros(meta.shape, dtype = np.float32)

        if data is None:
            data = np.zeros(meta.shape, dtype = np.float32)

        if meta is None:
            meta = Image_meta_3d(data.shape)

        if not isinstance(data, np.ndarray):
            raise TypeError('data must be a np.ndarray object')

        if data.shape != meta.shape:
            raise ValueError('data must has same shape with its meta')

        data.astype(np.float32)
        super().__init__(data, meta)

    def map(self, f):
        return Image_3d(*f(self.data, self.meta))
