import numpy as np
import tensorflow as tf
from doufo import multidispatch
from dxl.learn import Model

from srf.data import Image

__all__ = ['BackProjectionOrdinary', 'BackProjectionOrdinary_psf', 'backprojection',
           'MapOrdinary', 'map_lors']


class BackProjection(Model):
    def __init__(self, name = 'backprojection'):
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
                 name = 'backprojection_ordinary'):
        super().__init__(name)
        self.physical_model = physical_model

    def kernel(self, projection_data, image):
        return backprojection(self.physical_model, projection_data, image)

    @property
    def parameters(self):
        return []


class BackProjectionOrdinary_psf(BackProjection):
    """
    A unified backprojection entry.
    """

    def __init__(self,
                 physical_model,
                 psf_data,
                 name = 'backprojection_ordinary_psf'):
        super().__init__(name)
        self.physical_model = physical_model
        self.psf_data = psf_data

    def kernel(self, projection_data, image):
        bproj = backprojection(self.physical_model, projection_data, image)
        grid = image.data.shape.as_list()
        nx, ny, nz = grid[0], grid[1], grid[2]
        mat_xy = psf_mat_xy(self.psf_data)
        mat_z = psf_mat_z(self.psf_data)
        squ = tf.reshape(bproj.data, [nx * ny, nz])
        result1 = squ @ mat_z
        result2 = tf.sparse_tensor_dense_matmul(mat_xy, result1, adjoint_a = True)
        return Image(tf.reshape(result2, grid), image.center, image.size)

    @property
    def parameters(self):
        return []


@multidispatch(nargs = 3, nouts = 1)
def backprojection(physical_model, projection_data, image):
    # print("debug here !!!!!!!!!!!!!!!")
    # print(physical_model)
    raise NotImplementedError


class MapOrdinary(BackProjection):
    """
    A unified backprojection entry.
    """

    def __init__(self,
                 physical_model,
                 name = None):
        super().__init__(name)
        self.physical_model = physical_model

    def kernel(self, projection_data, image):
        return map_lors(self.physical_model, projection_data, image)

    @property
    def parameters(self):
        return []


@multidispatch(nargs = 3, nouts = 1)
def map_lors(physical_model, projection_data, image):
    return physical_model.map_lors(projection_data, image)


def psf_mat_xy(psf_data):
    row = psf_data.matrix_xy.nonzero()[0].astype(np.int64)
    col = psf_data.matrix_xy.nonzero()[1].astype(np.int64)
    data = psf_data.matrix_xy.data
    N = psf_data.image_meta.n_xy
    indices = np.concatenate((np.expand_dims(row, 1),
                              np.expand_dims(col, 1)), 1)
    return tf.SparseTensor(indices = indices, values = data, dense_shape = [N, N])


def psf_mat_z(psf_data):
    return tf.constant(psf_data.matrix_z.todense())
