from dxl.learn.core import Model, Tensor
# from dxl.learn.model.tor_recon import Projection, BackProjection
import tensorflow as tf
import numpy as np
import warnings
import os
from scipy import sparse

from ..preprocess import preprocess_sino 
# TF_ROOT = os.environ.get('TENSORFLOW_ROOT')
# op = tf.load_op_library(
#     TF_ROOT + '/bazel-bin/tensorflow/core/user_ops/tof_tor.so')
# warnings.warn(DeprecationWarning())

# projection = op.projection_gpu
# backprojection = op.backprojection_gpu
# calmatrix = op.calmatrix_gpu


class ReconStep(Model):
    class KEYS(Model.KEYS):
        class TENSOR(Model.KEYS.TENSOR):
            IMAGE = 'image'
            # PROJECTION = 'projection'
            SYSTEM_MATRIX = 'system_matrix'
            EFFICIENCY_MAP = 'efficiency_map'
            SINOGRAM = 'sinogram'

    def __init__(self, name, image, efficiency_map,
                 sinograms,
                 matrixs,
                 graph_info):
        # self.grid = np.array(grid, dtype=np.int32)
        # self.position = np.array(position, dtype=np.float32)
        # self.size = np.array(size, dtype=np.float32)
        # self.kernel_width = float(kernel_width)
        # self.tof_bin = float(tof_bin)
        # self.tof_sigma2 = float(tof_sigma2)
        super().__init__(
            name,
            {
                self.KEYS.TENSOR.IMAGE:
                image,
                self.KEYS.TENSOR.EFFICIENCY_MAP:
                efficiency_map,
                self.KEYS.TENSOR.SINOGRAM:
                sinograms,
                self.KEYS.TENSOR.SYSTEM_MATRIX:
                matrixs
            },
            graph_info=graph_info)

    def kernel(self, inputs):
        # the default order of the image is z-dominant(z,y,x)
        # for projection another two images are created.
        img = inputs[self.KEYS.TENSOR.IMAGE].data

        effmap = inputs[self.KEYS.TENSOR.EFFICIENCY_MAP].data

        sinos = inputs[self.KEYS.TENSOR.SINOGRAM].data
        matrixs = inputs[self.KEYS.TENSOR.SYSTEM_MATRIX].data
        # sess=tf.Session()
        # img_np = img.eval(session=sess)
        # effmap_np = effmap.eval(session=sess)
        # sinos_mp = sinos.eval(session=sess)
        # matrixs_mp = matrixs.eval(session=sess)


        """
        The following codes need rewrite
        """
        
        proj = tf.matmul(matrixs,img, a_is_sparse=True)
        con = tf.ones(proj.shape)/100000000
        proj = proj+con
        temp_proj = sinos/proj
        temp_bp = tf.matmul(matrixs,temp_proj,transpose_a=True,a_is_sparse=True) 
        result = img * effmap * temp_bp

        # result = imgz / (effmap+1e-8) * bpz
        # proj = matrixs_np.dot(img_np)
        # con = np.ones_like(proj)/100000
        # proj = np.add(proj, con)
        # temp_proj = sinos_np/proj
        # matrixs_transpose = np.transpose(matrixs_np)
        # temp_bp = matrixs_transpose.dot(temp_proj)
        # result_np = img_np *effmap_np *temp_bp
        # result = tf.convert_to_tensor(result_np)
        return Tensor(result, None, self.graph_info.update(name=None))

# class Calmatrix(Model):
#     class KEYS(Model.KEYS):
#         class TENSOR(Model.KEYS.TENSOR):
#             IMAGE = 'image'

#     def __init__(self,name,
#                  grid,position,size,
#                  graph_info):
#         self.grid = np.array(grid, dtype=np.int32)
#         self.position = np.array(position, dtype=np.float32)
#         self.size = np.array(size, dtype=np.float32)
#         super().__init__(
#             name,
#             graph_info=graph_info)
    
#     def kernel(self,inputs):
#         img = inputs[self.KEYS.TENSOR.IMAGE].data
#         grid = self.grid
#         position = self.position
#         size = self.size
#         system_matrix = calmatrix(
#             grid=grid,
#             position=position,
#             size=size,
#             model=model)
#         return Tensor(system_matrix,None,self.graph_info.update(name=None))



# class Projection(Model):
#     class KEYS(Model.KEYS):
#         class TENSOR(Model.KEYS.TENSOR):
#             IMAGE = 'image'
#             # PROJECTION = 'projection'
#             SYSTEM_MATRIX = 'system_matrix'
#             # EFFICIENCY_MAP = 'efficiency_map'
#             # LORS = 'lors'
#             # SINOGRAM = 'sinogram'

#     def __init__(self, name, image,
#                  system_matrix,
#                  graph_info):
#         # self.grid = np.array(grid, dtype=np.int32)
#         # self.position = np.array(position, dtype=np.float32)
#         # self.size = np.array(size, dtype=np.float32)
#         # self.kernel_width = float(kernel_width)
#         # self.tof_bin = float(tof_bin)
#         # self.tof_sigma2 = float(tof_sigma2)
#         # print(tof_bin)
#         super().__init__(
#             name,
#             {
#                 self.KEYS.TENSOR.IMAGE: image,
#                 self.KEYS.TENSOR.SYSTEM_MATRIX: system_matrix,
#             },
#             graph_info=graph_info)

#     def kernel(self, inputs):
#         img = inputs[self.KEYS.TENSOR.IMAGE].data
#         # grid = self.grid
#         # position = self.position
#         # size = self.size
#         # sinos = inputs[self.KEYS.TENSOR.SINOGRAM].data
#         matrix = inputs[self.KEYS.TENSOR.SYSTEM_MATRIX].data
#         projection_value = matrix*img
#         return Tensor(projection_value, None, self.graph_info.update(name=None))


# class BackProjection(Model):
#     class KEYS(Model.KEYS):
#         class TENSOR(Model.KEYS.TENSOR):
#             #IMAGE = 'image'
#             # PROJECTION = 'projection'
#             SYSTEM_MATRIX = 'system_matrix'
#             # EFFICIENCY_MAP = 'efficiency_map'
#             #LORS = 'lors'
#             SINOGRAMS = 'sinogram'

#     def __init__(self, name,
#                  sinos,
#                  system_matrix,
#                  graph_info):
#         # self.grid = np.array(grid, dtype=np.int32)
#         # self.position = np.array(position, dtype=np.float32)
#         # self.size = np.array(size, dtype=np.float32)
#         super().__init__(
#             name,
#             {
#                # self.KEYS.TENSOR.IMAGE: image,
#                 self.KEYS.TENSOR.SINOGRAMS: sinos,
#                 self.KEYS.TENSOR.SYSTEM_MATRIX: system_matrix
#             },
#             graph_info=graph_info)

#     def kernel(self, inputs):
#         # img = inputs[self.KEYS.TENSOR.IMAGE].data
#         # grid = self.grid
#         # position = self.position
#         # size = self.size
#         sinos = inputs[self.KEYS.TENSOR.SINOGRAMS].data
#         matrix = inputs[self.KEYS.TENSOR.SYSTEM_MATRIX].data
#         matrix_transpose = np.transpose(matrix)
#         backprojection_image = matrix_transpose * sinos
#         return Tensor(backprojection_image, None, self.graph_info.update(name=None))
