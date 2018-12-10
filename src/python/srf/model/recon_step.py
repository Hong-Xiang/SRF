from dxl.learn import Model
from srf.data import Image
from doufo import func, multidispatch

"""
ReconstructionStep is the abstract representation of 
'one step of medical image reconstruction',
It currently representing:
1. projection = Projection(Image, ProjectionDomain)
2. backprojection = Backprojection(projection::ProjectionData, ImageDomain)
3. image_next = image / efficiency_map * backprojection
"""

__all__ = ['ReconStep', 'PSFReconStep', 'mlem_update', 'mlem_update_normal']


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
        KT = self.KEYS.TENSOR
        image = inputs[KT.IMAGE]
        efficiency_map = inputs[KT.EFFICIENCY_MAP]
        projection_data = inputs[KT.PROJECTION_DATA]
        proj = self.projection(image, projection_data)
        # proj1 = type(proj)(proj.lors, proj.values / projection_data.values)
        back_proj = self.backprojection(proj, image)
        return self.update(image, back_proj, efficiency_map)

    @property
    def parameters(self):
        return []


def mlem_update(image_prev: Image, image_succ: Image, efficiency_map: Image):
    return image_prev.fmap(lambda d: d * efficiency_map.data * image_succ.data)


def mlem_update_normal(image_prev: Image, image_succ: Image, efficiency_map: Image):
    return image_prev.fmap(lambda d: d / efficiency_map.data * image_succ.data)


class PSFReconStep(ReconStep):
    class KEYS(ReconStep.KEYS):
        class TENSOR(ReconStep.KEYS.TENSOR):
            PSF_XY = 'psf_xy'
            PSF_Z = 'psf_z'
    def __init__(self, name, projection, backprojection, update):
        super().__init__(name, projection, backprojection, update)

    def kernel(self, inputs):
        KT = self.KEYS.TENSOR

        image = inputs[KT.IMAGE]
        efficiency_map = inputs[KT.EFFICIENCY_MAP]
        projection_data = inputs[KT.PROJECTION_DATA]
        psf_xy = inputs[KT.PSF_XY]
        psf_z = inputs[KT.PSF_Z]

        # here start the extra psf process,
        # the varabiles create below are tensorflow-like type.
        # TODO: rewrite by encapsulating in doufo and dxlearn.
        import tensorflow as tf
        grid = image.data.get_shape()
        print("the type of image grid is ",grid)
        image_vectorized = tf.reshape(image.data, [grid[0]*grid[1], grid[2]])
        print('psf xy shape:', psf_xy.data.get_shape())
        print('psf z shape:', psf_z.data.get_shape())
        print('image_vectorized shape:', image_vectorized.shape)
        # image_mul_psf_z = image_vectorized
        image_mul_psf_z = image_vectorized@tf.transpose(psf_z.data)
        
        image_psf_processed = tf.sparse_tensor_dense_matmul(
            psf_xy.data, image_mul_psf_z)
        # image_psf_processed = image_mul_psf_z
        image_psf_processed = tf.reshape(
            image_psf_processed, shape=tf.shape(image.data))
        image_psf = Image(image_psf_processed, image.center, image.size)
        # ####################

        # original ReconStep
        proj = self.projection(image_psf, projection_data)
        back_proj = self.backprojection(proj, image_psf)
        # from dxl.learn.tensor import transpose

        # here start the extra psf process,
        # the varabiles create below are tensorflow-like type.
        # TODO: rewrite by encapsulating in doufo and dxlearn.
        
        back_proj_vectorized = tf.reshape(back_proj.data, [grid[0]*grid[1], grid[2]])
        # back_proj_mul_psf_z  = back_proj_vectorized
        back_proj_mul_psf_z  = back_proj_vectorized@psf_z.data
        back_proj_psf_processed = tf.sparse_tensor_dense_matmul(
            tf.sparse_transpose(psf_xy.data), back_proj_mul_psf_z)
        # back_proj_psf_processed = back_proj_mul_psf_z
        back_proj_psf = tf.reshape(
            back_proj_psf_processed, shape=tf.shape(image.data))

        
        back_proj = Image(back_proj_psf, image.center, image.size)
        ######################################
        return self.update(image, back_proj, efficiency_map)
