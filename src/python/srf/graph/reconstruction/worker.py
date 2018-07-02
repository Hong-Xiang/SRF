from dxl.learn.core import Graph, Tensor, Variable, Constant, NoOp
from dxl.learn.distribute import DistributeGraphInfo, Host, Master, JOB_NAME

import numpy as np
import tensorflow as tf

from srf.tensor import Image
from dxl.learn.function import ControlDependencies


class WorkerGraph(Graph):
    class KEYS(Graph.KEYS):
        class CONFIG(Graph.KEYS.CONFIG):
            TASK_INDEX = 'task_index'

        class TENSOR(Graph.KEYS.TENSOR):
            X = 'x'
            TARGET = 'target'
            RESULT = 'result'
            INIT = 'init'
            UPDATE = 'update'

        class GRAPH(Graph.KEYS.GRAPH):
            RECONSTRUCTION = 'reconstruction'

    def __init__(self,
                 info,
                 recon_step,
                 x: Tensor=None,
                 x_target: Variable=None,
                 *,
                 loader=None,
                 task_index=None,
                 center,
                 size):
        super().__init__(info, tensors={
            self.KEYS.TENSOR.X: x,
            self.KEYS.TENSOR.TARGET: x_target
        }, graphs={
            self.KEYS.GRAPH.RECONSTRUCTION: recon_step
        }, config={
            self.KEYS.CONFIG.TASK_INDEX: task_index,
            'center': center,
            'size': size
        })
        self._loader = loader

    def kernel(self, inputs=None):
        inputs = self._construct_inputs()
        result = self._construct_x_result(inputs)
        self._construct_x_update(result)

    @property
    def task_index(self):
        return self.config('task_index')

    def _construct_inputs(self):
        KT = self.KEYS.TENSOR
        with tf.variable_scope('local_inputs'):
            local_inputs, local_inputs_init = self._loader.load(self)
            for k, v in local_inputs.items():
                self.tensors[k] = v
            with ControlDependencies(local_inputs_init):
                self.tensors[self.KEYS.TENSOR.INIT] = NoOp()
        inputs = {'image': Image(self.tensors[KT.X], self.config('center'),
                                 self.config('size')),
                  KT.TARGET: self.tensors[KT.TARGET]}
        inputs.update(local_inputs)
        return inputs

    def _construct_x_result(self, inputs):
        KS, KT = self.KEYS.GRAPH, self.KEYS.TENSOR
        result = self.tensors[KT.RESULT] = self.graphs[KS.RECONSTRUCTION](
            inputs)
        return result

    def _construct_x_update(self, result):
        """
        update the master x buffer with the x_result of workers.
        """
        KT = self.KEYS.TENSOR
        self.tensors[KT.UPDATE] = self.tensors[KT.TARGET].assign(
            self.tensors[KT.RESULT])


class OSEMWorkerGraph(WorkerGraph):
    class KEYS(WorkerGraph.KEYS):
        class CONFIG(WorkerGraph.KEYS.CONFIG):
            NB_SUBSETS = 'nb_subsets'

        class TENSOR(WorkerGraph.KEYS.TENSOR):
            SUBSET = 'subset'

    def __init__(self, info, x, x_target, subset, *, recon_step, loader, nb_subsets=None):
        super().__init__(info, x, x_target, loader=loader, recon_step=recon_step)
        self.tensors.update({self.KEYS.TENSOR.SUBSET: subset})
        self.config.update({self.KEYS.CONFIG.NB_SUBSETS: nb_subsets})

    def _construct_inputs(self):
        KC, KT = self.KEYS.CONFIG, self.KEYS.TENSOR
        inputs = super()._construct_inputs()
        for k in self._loader.to_split(self):
            inputs[k] = inputs[k].split_with_index(
                self.config(KC.NB_SUBSETS), self.tensor(KT.SUBSET))
        return inputs
