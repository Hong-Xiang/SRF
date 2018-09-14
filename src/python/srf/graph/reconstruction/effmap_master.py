import numpy as np
import tensorflow as tf

from dxl.learn.core import Graph, Tensor, NoOp, Variable
from dxl.learn.model import Summation
from doufo.tensor import sum_
from dxl.learn.tensor import assign


class MasterGraph(Graph):
    class KEYS(Graph.KEYS):
        class CONFIG(Graph.KEYS.CONFIG):
            NB_WORKERS = 'nb_workers'

        class TENSOR(Graph.KEYS.TENSOR):
            X = 'x'
            BUFFER = 'x_buffer'
            UPDATE = 'x_update'
            INIT = 'init'

        class GRAPH(Graph.KEYS.GRAPH):
            SUMMATION = 'summation'

    def __init__(self, info, *, config=None, loader=None, nb_workers=None):
        """
        Args:
            info: the graph info
            config:
            loader: the data loader
            nb_workers: number of the workers
        """
        super().__init__(info, config=config)

    def kernel(self):
        self._construct_x()
        self._construct_init()
        self._construct_summation()

    @property
    def nb_workers(self):
        return self.config(self.KEYS.CONFIG.NB_WORKERS)

    def _construct_x(self):
        KT, KC = self.KEYS.TENSOR, self.KEYS.CONFIG
        x = self.tensors[KT.X] = Variable(self.info.child_tensor(KT.X),
                                          initializer=self._loader_load(self))
        self.tensors[KT.BUFFER] = [
            Variable(self.info.child_tensor(
                '{}_{}'.format(KT.BUFFER, i)),
                shape=x.shape,
                dtype=x.dtype) for i in range(self.config(KC.NB_WORKERS))
        ]

    def _construct_init(self):
        KT = self.KEYS.TENSOR
        to_init = [self.get_or_create_tensor(
            KT.X)] + self.get_or_create_tensor(KT.BUFFER)
        with tf.control_dependencies([t.init().data for t in to_init]):
            self.tensors[KT.INIT] = NoOp()

    def _construct_summation(self):
        KT, KS = self.KEYS.TENSOR, self.KEYS.GRAPH
        x_s = sum_(self.tensors[KT.BUFFER], axis=0)
        self.tensors[KT.UPDATE] = assign(self.tensors[KT.X], x_s)
