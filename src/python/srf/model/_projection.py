import numpy as np
import tensorflow as tf
from doufo import multidispatch
from dxl.learn import Model

from srf.data import Image

__all__ = ['ProjectionOrdinary', 'ProjectionOrdinary_psf', 'projection']


class Projection(Model):
    def __init__(self, name):
        super().__init__(name)

    def build(self, *args):
        pass

    def kernel(self, image, projection_data):
        raise NotImplementedError


class ProjectionOrdinary(Projection):
    """
    A unified projection entry.
    """

    def __init__(self,
                 physical_model,
                 name = 'projection_ordinary'):
        super().__init__(name)
        self.physical_model = physical_model

    def kernel(self, image, projection_data):
        return projection(self.physical_model, image, projection_data)

    @property
    def parameters(self):
        return []


class ProjectionOrdinary_psf(Projection):
    """
    A unified projection entry.
    """

    def __init__(self,
                 physical_model,
                 psf_data,
                 name = 'projection_ordinary_psf'):
        super().__init__(name)
        self.physical_model = physical_model
        self.psf_data = psf_data

    def kernel(self, image, projection_data):
        grid = image.data.shape.as_list()
        nx, ny, nz = grid[0], grid[1], grid[2]
        mat_xy = psf_mat_xy(self.psf_data)
        mat_z = psf_mat_z(self.psf_data)
        squ = tf.reshape(image.data, [nx * ny, nz])
        result1 = squ @ tf.transpose(mat_z)
        result2 = tf.sparse_tensor_dense_matmul(mat_xy, result1, adjoint_a = True)
        image_psf = Image(tf.reshape(result2, grid), image.center, image.size)
        return projection(self.physical_model, image_psf, projection_data)

    @property
    def parameters(self):
        return []

@multidispatch(nargs = 3, nouts = 1)
def projection(physics, image, projection_data):
    raise NotImplementedError(
        f"No projection implementation for physics model: {type(physics)}; image: {type(image)}; projection_data: {type(projection_data)}.")


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
