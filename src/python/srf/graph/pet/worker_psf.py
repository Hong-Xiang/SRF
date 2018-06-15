# from .utils import constant_tensor
from dxl.learn.core import MasterHost, Graph, Tensor, tf_tensor, variable, Constant, NoOp
from dxl.learn.core import DistributeGraphInfo, Host
from dxl.learn.core.distribute import JOB_NAME
from dxl.learn.core.tensor import SparseMatrix
import numpy as np
import tensorflow as tf

from srf.graph.pet.worker import WorkerGraphBase
from scipy import sparse


class WorkerGraphPSF(WorkerGraphBase):
    class KEYS(WorkerGraphBase.KEYS):
        class TENSOR(WorkerGraphBase.KEYS.TENSOR):
            LORS = 'lors'
            ASSIGN_LORS = 'assign_lors'
            PSF_XY = 'psf_xy'
            PSF_Z = 'psf_z'

            X_PSF = 'x_psf'
            IMAGE_PSF = 'image_psf'

        class CONFIG(WorkerGraphBase.KEYS.CONFIG):
            KERNEL_WIDTH = 'kernel_width'
            IMAGE_INFO = 'image'
            TOF_BIN = 'tof_bin'
            TOF_SIGMA2 = 'tof_sigma2'
            TOF_RES = 'tof_res'
            LORS_INFO = 'lors_info'
            TOF = 'tof'

        class SUBGRAPH(WorkerGraphBase.KEYS.SUBGRAPH):
            RECON_STEP = 'recon_step'

    AXIS = ('x', 'y', 'z')
    REQURED_INPUTS = (KEYS.TENSOR.EFFICIENCY_MAP,
                      KEYS.TENSOR.LORS, KEYS.TENSOR.PSF_XY, KEYS.TENSOR.PSF_Z)

    def __init__(self,
                 x, x_target, subset, inputs, task_index,
                 nb_subsets=1,
                 kernel_width=None,
                 image_info=None,
                 tof_bin=None,
                 tof_sigma2=None,
                 lors_info=None,
                 graph_info=None,
                 name=None):
        super().__init__(x, x_target, subset, inputs, task_index=task_index, name=name,
                         config={
                             self.KEYS.CONFIG.KERNEL_WIDTH: kernel_width,
                             self.KEYS.CONFIG.IMAGE_INFO: image_info,
                             self.KEYS.CONFIG.TOF_BIN: tof_bin,
                             self.KEYS.CONFIG.TOF_SIGMA2: tof_sigma2,
                             self.KEYS.CONFIG.LORS_INFO: lors_info,
                         }, graph_info=graph_info)

    def _construct_inputs(self, inputs):
        KT = self.KEYS.TENSOR
        effmap_name = KT.EFFICIENCY_MAP

        super()._construct_inputs({effmap_name: inputs[effmap_name]})
        self.tensors[KT.PSF_XY], self.tensors[KT.PSF_Z] = self.process_psf(
            inputs[KT.PSF_XY], inputs[KT.PSF_Z])

        self.tensors[KT.LORS] = {a: self.process_lors(
            inputs[KT.LORS][a], a) for a in self.AXIS}
        self.tensors[KT.INIT] = NoOp()

    def process_lors(self, lor: np.ndarray, axis):
        KT = self.KEYS.TENSOR
        KC = self.KEYS.CONFIG
        step = lor.shape[0] // self.config(KC.NB_SUBSETS)
        # step = lor.shape[0]
        columns = lor.shape[1]
        lor = Constant(lor, None, self.info.child(KT.LORS))
        if self.config(KC.NB_SUBSETS) == 1:
            return lor
        lor = tf.slice(lor.data,
                       [self.tensor(KT.SUBSET).data * step, 0],
                       [step, columns])
        return Tensor(lor, None, self.info.child('{}_{}'.format(KT.LORS, axis)))

    def process_psf(self, psf_xy: sparse.coo_matrix, psf_z: np.ndarray):
        '''
        create the PSF kernels and its transpose.
        '''
        KT = self.KEYS.TENSOR
        psf_xy = SparseMatrix(psf_xy, None, self.info.child(KT.PSF_XY))
        psf_z = Constant(psf_z, None, self.info.child(KT.PSF_Z))
        return psf_xy, psf_z

    def _construct_x_result(self):
        KT = self.KEYS.TENSOR
        KC = self.KEYS.CONFIG
        from ...model.psf_step import PSFStep
        # print(self.config('tof')[KC.TOF_SIGMA2])

        grid = self.config(KC.IMAGE_INFO)['grid']
        x = self.tensor(KT.X).data
        x_reshaped = tf.reshape(x, [grid[0]*grid[1], grid[2]])
        print('psf shape:', self.tensor(KT.PSF_XY).data.get_shape())
        print('x shape:', x_reshaped.shape)
        x_mul_z = x_reshaped@tf.transpose(self.tensor(KT.PSF_Z).data)
        x_psf = tf.sparse_tensor_dense_matmul(
            self.tensor(KT.PSF_XY).data, x_mul_z)
        x_psf = tf.reshape(x_psf, shape=tf.shape(self.tensor(KT.X).data))
        self.tensors[KT.X_PSF] = Tensor(x_psf, None, self.info.child(KT.X_PSF))

        self.subgraphs[self.KEYS.SUBGRAPH.RECON_STEP] = PSFStep(
            self.name / 'recon_step_{}'.format(self.task_index),
            self.tensor(KT.X_PSF),
            self.config(KC.IMAGE_INFO)['grid'],
            self.config(KC.IMAGE_INFO)['center'],
            self.config(KC.IMAGE_INFO)['size'],
            self.config('tor')[KC.KERNEL_WIDTH],
            self.config('tof')[KC.TOF_BIN],
            self.config('tof')[KC.TOF_SIGMA2],
            self.tensor(KT.LORS)['x'],
            self.tensor(KT.LORS)['y'],
            self.tensor(KT.LORS)['z'],
            self.info.update(name=None))
        img_res = self.subgraph(self.KEYS.SUBGRAPH.RECON_STEP)()

        img_reshaped = tf.reshape(img_res.data,  [grid[0]*grid[1], grid[2]])
        img_mul_z = img_reshaped@self.tensor(KT.PSF_Z).data

        img_trans = tf.sparse_tensor_dense_matmul(
            tf.sparse_transpose(self.tensor(KT.PSF_XY).data), img_mul_z)
        img_psf = tf.reshape(img_trans, shape=tf.shape(self.tensor(KT.X).data))
        self.tensors[KT.IMAGE_PSF] = Tensor(
            img_psf, None, self.info.child(KT.IMAGE_PSF))
        x = self.tensor(KT.X).data
        # the effmap should be processed by psf.
        effmap = self.tensor(KT.EFFICIENCY_MAP).data
        image_psf = self.tensor(KT.IMAGE_PSF).data
        x_res = x/effmap*image_psf
        x_res = Tensor(x_res, None, self.info.child(KT.IMAGE_PSF))
        self.tensors[KT.RESULT] = x_res
