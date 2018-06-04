# from .utils import constant_tensor
from dxl.learn.core import MasterHost, Graph, Tensor, tf_tensor, variable, Constant, NoOp
from dxl.learn.core import DistributeGraphInfo, Host
from dxl.learn.core.distribute import JOB_NAME
import numpy as np
import tensorflow as tf


class OsemWorkerGraph(Graph):
    class KEYS(Graph.KEYS):
        class CONFIG(Graph.KEYS.CONFIG):
            TASK_INDEX = 'task_index'
            NB_SUBSETS = 'nb_subsets'

        class TENSOR(Graph.KEYS.TENSOR):
            X = 'x'
            SUBSET = 'subset'
            TARGET = 'target'
            RESULT = 'result'
            EFFICIENCY_MAP = 'efficiency_map'
            INIT = 'init'
            UPDATE = 'update'

    REQURED_INPUTS = (KEYS.TENSOR.EFFICIENCY_MAP,)

    def __init__(self, x, x_target, subset,
                 inputs,
                 name=None,
                 graph_info=None,
                 config=None,
                 *,
                 task_index=None,
                 nb_subsets=None):
        if name is None:
            name = 'worker_graph_{}'.format(task_index)
        if config is None:
            config = {}
        config.update({
            self.KEYS.CONFIG.TASK_INDEX: task_index,
            self.KEYS.CONFIG.NB_SUBSETS: nb_subsets,
        })
        inputs = {k: v for k, v in inputs.items() if k in self.REQURED_INPUTS}
        super().__init__(name, graph_info=graph_info, tensors={
            self.KEYS.TENSOR.X: x,
            self.KEYS.TENSOR.TARGET: x_target,
            self.KEYS.TENSOR.SUBSET: subset
        }, config=config)
        with self.info.variable_scope():
            with tf.name_scope('inputs'):
                self._construct_inputs(inputs)
            with tf.name_scope(self.KEYS.TENSOR.RESULT):
                self._construct_x_result()
            with tf.name_scope(self.KEYS.TENSOR.UPDATE):
                self._construct_x_update()

    @classmethod
    def default_config(cls):
        return {
            cls.KEYS.CONFIG.NB_SUBSETS: 1,
        }

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
        """
        divide x by the effciency map. 
        """
        KT = self.KEYS.TENSOR
        self.tensors[KT.RESULT] = self.tensor(
            KT.X) / self.tensor(KT.EFFICIENCY_MAP)

    def _construct_x_update(self):
        """
        update the master x buffer with the x_result of workers.
        """
        KT = self.KEYS.TENSOR
        self.tensors[KT.UPDATE] = self.tensor(
            KT.TARGET).assign(self.tensor(KT.RESULT))
