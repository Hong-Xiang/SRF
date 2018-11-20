import tensorflow as tf
from doufo import multidispatch
from dxl.learn import Model

from srf.data import Image

__all__ = ['ProjectionOrdinary', 'projection']


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
        if True:
            grid = image.data.shape.as_list()
            nx, ny, nz = grid[0], grid[1], grid[2]
            mat_xy = psf_mat_xy()
            mat_z = psf_mat_z()
            squ = tf.reshape(image.data, [nx * ny, nz])
            # result1 = tf.sparse_tensor_dense_matmul(mat_z, tf.transpose(squ))
            result1 = squ @ tf.transpose(mat_z)
            result2 = tf.sparse_tensor_dense_matmul(mat_xy, result1, adjoint_a = True)
            image_psf = Image(tf.reshape(result2, grid), image.center, image.size)
            return projection(self.physical_model, image_psf, projection_data)
        else:
            return projection(self.physical_model, image, projection_data)

    @property
    def parameters(self):
        return []


@multidispatch(nargs = 3, nouts = 1)
def projection(physics, image, projection_data):
    raise NotImplementedError(
        f"No projection implementation for physics model: {type(physics)}; image: {type(image)}; projection_data: {type(projection_data)}.")


def psf_mat_xy():
    from srf.psf.data.psf import PSF_3d
    import tensorflow as tf
    import h5py
    import numpy as np
    filename = '/home/bill52547/Workspace/SRF_new_start/run/psf_2m/mat_psf.h5'
    fin = h5py.File(filename, 'r')
    row = np.array(fin['PSF']['_matrix_xy_row'][:], dtype = np.int64)
    col = np.array(fin['PSF']['_matrix_xy_col'][:], dtype = np.int64)
    data = np.array(fin['PSF']['_matrix_xy_data'][:], dtype = np.float32)
    psf = PSF_3d.load_h5(filename)
    N = psf.image_meta.n_xy
    indices = np.concatenate((np.expand_dims(row, 1),
                              np.expand_dims(col, 1)), 1)
    return tf.SparseTensor(indices = indices, values = data, dense_shape = [N, N])


def psf_mat_z():
    from srf.psf.data.psf import PSF_3d
    filename = '/home/bill52547/Workspace/SRF_new_start/run/psf_2m/mat_psf.h5'
    psf = PSF_3d.load_h5(filename)
    return tf.constant(psf.matrix_z.todense())
