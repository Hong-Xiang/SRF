import tensorflow as tf
from dxl.learn import Model

from srf.data import Image

"""
ReconstructionStep is the abstract representation of 
'one step of medical image reconstruction',
It currently representing:
1. projection = Projection(Image, ProjectionDomain)
2. backprojection = Backprojection(projection::ProjectionData, ImageDomain)
3. image_next = image / efficiency_map * backprojection
"""

__all__ = ['ReconStep', 'mlem_update', 'mlem_update_normal']


class ReconStep(Model):
    class KEYS(Model.KEYS):
        class TENSOR:
            IMAGE = 'image'
            EFFICIENCY_MAP = 'efficiency_map'
            PROJECTION_DATA = 'projection_data'

    def __init__(self, name, projection, backprojection, update):
        super().__init__(name)
        self.projection = projection
        self.backprojection = backprojection
        self.update = update

    def build(self, *args):
        pass

    def kernel(self, inputs):
        image = inputs[self.KEYS.TENSOR.IMAGE]
        efficiency_map = inputs[self.KEYS.TENSOR.EFFICIENCY_MAP]
        projection_data = inputs[self.KEYS.TENSOR.PROJECTION_DATA]
        proj = self.projection(image, projection_data)
        # proj1 = type(proj)(proj.lors, proj.values / projection_data.values)
        back_proj = self.backprojection(proj, image)
        return self.update(image, back_proj, efficiency_map)
        # grid = image.data.shape.as_list()
        # nx, ny, nz = grid[0], grid[1], grid[2]
        # mat_xy = psf_mat_xy()
        # mat_z = psf_mat_z()
        # squ = tf.reshape(back_proj.data, [nx * ny, nz])
        # result1 = squ @ mat_z
        # result2 = tf.sparse_tensor_dense_matmul(mat_xy, result1, adjoint_a = True)
        # effmap_psf = Image(tf.reshape(result2, grid), image.center, image.size)
        #
        # return self.update(image, back_proj, effmap_psf)
    @property
    def parameters(self):
        return []


def mlem_update(image_prev: Image, image_succ: Image, efficiency_map: Image):
    return image_prev.fmap(lambda d: d * efficiency_map.data * image_succ.data)


def mlem_update_normal(image_prev: Image, image_succ: Image, efficiency_map: Image):
    return image_prev.fmap(lambda d: d / efficiency_map.data * image_succ.data)


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
