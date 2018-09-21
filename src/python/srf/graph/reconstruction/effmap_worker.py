from dxl.learn.core import Graph, Tensor, Variable, Constant, NoOp

import numpy as np
import tensorflow as tf
from srf.data import Image


class WorkerGraph(Graph):
    """Base class of a worker graph to compute efficiency map. 
    
    """
    class KEYS(Graph.KEYS):
        class CONFIG(Graph.KEYS.CONFIG):
            TASK_INDEX = 'task_index'

        class TENSOR(Graph.KEYS.TENSOR):
            X = 'x'
            TARGET = 'target'
            INIT = 'init'
            RESULT = 'result'

        class GRAPH(Graph.KEYS.GRAPH):
            EFFMAP = 'effmap'

    def __init__(self,
                 info,
                 x: Tensor,
                 x_target: Variable,
                 *,
                 effmap_step=None,
                 loader=None,
                 tensors=None,
                 graphs=None,
                 config=None):
        self._loader = loader
        if tensors is None:
            tensors = {}
        tensors = dict(tensors)
        tensors.update({
            self.KEYS.TENSOR.X: x,
            self.KEYS.TENSOR.TARGET: x_target,
        })
        if graphs is None:
            graphs = {}
        if effmap_step is not None:
            graphs.update({
                self.KEYS.GRAPH.EFFMAP: effmap_step})
        super().__init__(info,
                         tensors=tensors,
                         config=config,
                         graphs=graphs)

    def kernel(self):
        inputs = self._consturct_inputs()
        self._consturct_x_result(inputs)

    @property
    def task_index(self):
        return self.config('task_index')

    def _construrt_inputs(self):
        KT = self.KEYS.TENSOR
        with tf.variable_scope('local_inputs'):
            local_inputs, local_inputs_init = self._loader.load(self)
            for k, v in local_inputs.items():
                self.tensors[k] = v
            with tf.control_dependencies([t.data for t in local_inputs_init]):
                self.tensors[self.KEYS.TENSOR.INIT] = NoOp()
        inputs = {'image': Image(self.tensor(KT.X),
                                 self.config('center'),
                                 self.config('size')),
                  KT.TARGET: self.tensor(KT.TARGET)}
        inputs.update(local_inputs)
        return inputs

    def _construct_x_result(self, inputs):
        KS, KT = self.KEYS.GRAPH, self.KEYS.TENSOR
        compute_map = self.graphs[KS.EFFMAP]
        result = compute_map(inputs)
        self.tensors[KT.RESULT] = result
        return result

