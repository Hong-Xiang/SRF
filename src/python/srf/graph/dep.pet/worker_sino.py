# from .utils import constant_tensor
from dxl.learn.core import MasterHost, Graph, Tensor, tf_tensor, variable, Constant, NoOp 
from dxl.learn.core.tensor import SparseMatrix
from dxl.learn.core import DistributeGraphInfo, Host
from dxl.learn.core.distribute import JOB_NAME
import numpy as np
import tensorflow as tf

from ...model.sinogram import ReconStep


class WorkerGraphBase(Graph):
    class KEYS(Graph.KEYS):
        class CONFIG(Graph.KEYS.CONFIG):
            TASK_INDEX = 'task_index'
            #NB_SUBSETS = 'nb_subsets'

        class TENSOR(Graph.KEYS.TENSOR):
            X = 'x'
            #SUBSET = 'subset'
            TARGET = 'target'
            RESULT = 'result'
            EFFICIENCY_MAP = 'efficiency_map'
            INIT = 'init'
            UPDATE = 'update'

    REQURED_INPUTS = (KEYS.TENSOR.EFFICIENCY_MAP,)

    def __init__(self, x, x_target,
                 inputs,
                 name=None,
                 graph_info=None,
                 config=None,
                 *,
                 task_index=None):
        if name is None:
            name = 'worker_graph_{}'.format(task_index)
        if config is None:
            config = {}
        config.update({
            self.KEYS.CONFIG.TASK_INDEX: task_index,
        })
        inputs = {k: v for k, v in inputs.items() if k in self.REQURED_INPUTS}
        super().__init__(name, graph_info=graph_info, tensors={
            self.KEYS.TENSOR.X: x,
            self.KEYS.TENSOR.TARGET: x_target,
            #self.KEYS.TENSOR.SUBSET: subset
        }, config=config)
        with self.info.variable_scope():
            with tf.name_scope('inputs'):
                self._construct_inputs(inputs)
            with tf.name_scope(self.KEYS.TENSOR.RESULT):
                self._construct_x_result()
            with tf.name_scope(self.KEYS.TENSOR.UPDATE):
                self._construct_x_update()

    # @classmethod
    # def default_config(cls):
    #     return {
    #         cls.KEYS.CONFIG.NB_SUBSETS: 1,
    #     }

    def default_info(self):
        return DistributeGraphInfo(self.name, self.name, None, Host(JOB_NAME.WORKER, self.config(self.KEYS.CONFIG.TASK_INDEX)))

    @property
    def task_index(self):
        return self.config('task_index')

    def _construct_inputs(self, inputs):
        for k, v in inputs.items():
            self.tensors[k] = Constant(v, None, self.info.child(k))
        self.tensors[self.KEYS.TENSOR.INIT] = NoOp()

    def _construct_x_result(self):
        self.tensors[self.KEYS.TENSOR.RESULT] = self.tensor(
            self.KEYS.TENSOR.X) / self.tensor(self.KEYS.TENSOR.EFFICIENCY_MAP)

    def _construct_x_update(self):
        """
        update the master x buffer with the x_result of workers.
        """
        KT = self.KEYS.TENSOR
        self.tensors[KT.UPDATE] = self.tensor(
            KT.TARGET).assign(self.tensor(KT.RESULT))


class WorkerGraphSINO(WorkerGraphBase):
    class KEYS(WorkerGraphBase.KEYS):
        class TENSOR(WorkerGraphBase.KEYS.TENSOR):
            SINOS = 'sino'
            MATRIX = 'matrix'

        class CONFIG(WorkerGraphBase.KEYS.CONFIG):
            IMAGE_INFO = 'image'
            
        class GRAPH(WorkerGraphBase.KEYS.GRAPH):
            RECON_STEP = 'recon_step'

    REQURED_INPUTS = (KEYS.TENSOR.EFFICIENCY_MAP, KEYS.TENSOR.SINOS, KEYS.TENSOR.MATRIX)

    def __init__(self,
                 x, x_target, inputs, task_index,
                 image_info=None,
                 graph_info=None,
                 name=None):
        super().__init__(x, x_target, inputs, task_index=task_index, name=name,
                         config={
                             self.KEYS.CONFIG.IMAGE_INFO: image_info,
                         }, graph_info=graph_info)

    def _construct_inputs(self, inputs):
        KT = self.KEYS.TENSOR
        effmap_name = KT.EFFICIENCY_MAP
        super()._construct_inputs({effmap_name: inputs[effmap_name]})
        self.tensors[KT.SINOS] = self.process_sinos(inputs[KT.SINOS])
        self.tensors[KT.MATRIX] = self.process_matrix(inputs[KT.MATRIX]) 
        self.tensors[KT.INIT] = NoOp()

    def process_sinos(self, sino: np.ndarray):
        KT = self.KEYS.TENSOR
        sino = Constant(sino, None, self.info.child(KT.SINOS))
        return sino
    
    def process_matrix(self, matrix):
        KT = self.KEYS.TENSOR
        matrix = SparseMatrix(matrix, None, self.info.child(KT.MATRIX))
        return matrix

    def _construct_x_result(self):
        KT = self.KEYS.TENSOR
        # self.tensors[KT.RESULT] = self.tensor(KT.X)
        # return
        KC = self.KEYS.CONFIG
        self.graphs[self.KEYS.GRAPH.RECON_STEP] = ReconStep(
            self.name / 'recon_step_{}'.format(self.task_index),
            self.tensor(KT.X, is_required=True),
            self.tensor(KT.EFFICIENCY_MAP, is_required=True),
            self.tensor(KT.SINOS, is_required=True),
            self.tensor(KT.MATRIX, is_required=True),
            self.info.update(name=None))
        x_res = self.subgraph(self.KEYS.GRAPH.RECON_STEP)()
        self.tensors[KT.RESULT] = x_res
