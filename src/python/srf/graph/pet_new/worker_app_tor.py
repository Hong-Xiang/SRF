import numpy as np
import tensorflow as tf
from dxl.learn.core import MasterHost, Graph, Tensor, tf_tensor, variable, Constant, NoOp
from dxl.learn.core import DistributeGraphInfo, Host
from dxl.learn.core.distribute import JOB_NAME
from srf.graph.pet_new.worker_osem import WorkerGraphOsemBase
from srf.model.pet_new.tor_tof import TorStep


class WorkerGraphToR(WorkerGraphOsemBase):
    class KEYS(WorkerGraphOsemBase.KEYS):
        class TENSOR(WorkerGraphOsemBase.KEYS.TENSOR):
            LORS = 'lors'
            ASSIGN_LORS = 'ASSIGN_LORS'

        class CONFIG(WorkerGraphOsemBase.KEYS.CONFIG):
            KERNEL_WIDTH = 'kernel_width'
            IMAGE_INFO = 'image'
            TOF_BIN = 'tof_bin'
            TOF_SIGMA2 = 'tof_sigma2'
            TOF_RES = 'tof_res'
            LORS_INFO = 'lors_info'
            TOF = 'tof'

        class SUBGRAPH(WorkerGraphOsemBase.KEYS.SUBGRAPH):
            RECON_STEP = 'recon_step'

    AXIS = ('x', 'y', 'z')
    REQURED_INPUTS = (KEYS.TENSOR.EFFICIENCY_MAP, KEYS.TENSOR.LORS)

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
        self.tensors[KT.LORS] = {a: self.process_lors(
            inputs[KT.LORS][a], a) for a in self.AXIS}
        self.tensors[KT.INIT] = NoOp()

    def process_lors(self, lor: np.ndarray, axis):
        """
        """
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

    def _construct_x_result(self):
        KT = self.KEYS.TENSOR
        # self.tensors[KT.RESULT] = self.tensor(KT.X)
        # return
        KC = self.KEYS.CONFIG

        # print(self.config('tof')[KC.TOF_SIGMA2])

        self.subgraphs[self.KEYS.SUBGRAPH.RECON_STEP] = TorStep(
            self.name / 'recon_step_{}'.format(self.task_index),
            self.tensor(KT.X, is_required=True),
            self.tensor(KT.EFFICIENCY_MAP, is_required=True),
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
        x_res = self.subgraph(self.KEYS.SUBGRAPH.RECON_STEP)()
        self.tensors[KT.RESULT] = x_res
