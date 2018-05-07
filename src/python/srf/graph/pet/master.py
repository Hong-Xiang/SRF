from typing import Iterable

import numpy as np
import tensorflow as tf

from dxl.learn.core import Graph, DistributeGraphInfo, MasterHost, Tensor, NoOp
from dxl.learn.core.tensor import variable
from dxl.learn.core.utils import logger, map_data
from dxl.learn.model.on_collections import Summation
# from .utils import constant_tensor, variable_tensor


class MasterGraph(Graph):
    class KEYS(Graph.KEYS):
        class CONFIG(Graph.KEYS.CONFIG):
            NB_WORKERS = 'nb_workers'
            RENORMALIZATION = 'renormalization'
            NB_SUBSETS = 'nb_subsets'

        class TENSOR(Graph.KEYS.TENSOR):
            X = 'x'
            BUFFER = 'x_buffer'
            UPDATE = 'x_update'
            INIT = 'init'
            SUBSET = 'subset'
            INC_SUBSET = 'inc_subset'

        class SUBGRAPH(Graph.KEYS.SUBGRAPH):
            SUMMATION = 'summation'

    @classmethod
    def default_config(self):
        return {
            self.KEYS.CONFIG.NB_SUBSETS: 1,
            self.KEYS.CONFIG.RENORMALIZATION: False
        }

    def __init__(self, x, nb_workers=None, nb_subsets=None, name='master', graph_info=None):
        """
        `x`: numpy.ndarray
        """
        if graph_info is None:
            graph_info = DistributeGraphInfo(name, name, MasterHost.host())
        super().__init__(name=name,
                         graph_info=graph_info,
                         config={self.KEYS.CONFIG.NB_WORKERS: nb_workers,
                                 self.KEYS.CONFIG.NB_SUBSETS: nb_subsets})
        with self.info.variable_scope():
            self._construct_x(x)
            self._construct_subset()
            self._construct_init()
            self._construct_summation()
        # self._debug_info()

    def _construct_x(self, x):
        x = variable(self.info.child(self.KEYS.TENSOR.X), initializer=x)
        buffer = [
            variable(
                self.info.child('{}_{}'.format(self.KEYS.TENSOR.BUFFER, i)),
                shape=x.shape,
                dtype=x.dtype) for i in range(self.config(self.KEYS.CONFIG.NB_WORKERS))
        ]
        self.tensors[self.KEYS.TENSOR.X] = x
        self.tensors[self.KEYS.TENSOR.BUFFER] = buffer

    def _construct_subset(self):
        subset = variable(self.info.child(
            self.KEYS.TENSOR.SUBSET), initializer=0)
        self.tensors[self.KEYS.TENSOR.SUBSET] = subset
        with tf.name_scope(self.KEYS.TENSOR.INC_SUBSET):
            self.tensors[self.KEYS.TENSOR.INC_SUBSET] = subset.assign(
                (subset.data + 1) % self.config(self.KEYS.CONFIG.NB_SUBSETS))

    def _construct_init(self):
        KT = self.KEYS.TENSOR
        with tf.control_dependencies([self.tensor(KT.X).init().data, self.tensor(KT.SUBSET).init().data]):
            self.tensors[self.KEYS.TENSOR.INIT] = NoOp()

    def _construct_summation(self):
        gs = tf.train.get_or_create_global_step()
        gsa = gs.assign(gs + 1)
        KT, KG = self.KEYS.TENSOR, self.KEYS.SUBGRAPH
        self.subgraphs[KG.SUMMATION] = Summation(
            self.name / KG.SUMMATION, self.info.update(name=self.name / KG.SUMMATION, variable_scope=self.info.scope.name + '/' + KG.SUMMATION))
        x_s = self.subgraph(KG.SUMMATION)(self.tensor(KT.BUFFER))
        if self.config(self.KEYS.CONFIG.RENORMALIZATION):
            sum_s = tf.reduce_sum(x_s.data)
            sum_x = tf.reduce_sum(self.tensor(TK.X).data)
            x_s = x_s.data / sum_s * sum_x
        x_u = self.tensor(KT.X).assign(x_s)
        with tf.control_dependencies([x_u.data, self.tensor(KT.INC_SUBSET).data, gsa]):
            self.tensors[KT.UPDATE] = NoOp()
        return x_u

    # def _debug_info(self):
    #     logger.debug('Master graph constructed.')
    #     logger.debug('X: {}'.format(self.tensor(self.KEYS.TENSOR.X).data))
    #     logger.debug('BUFFER: {}'.format(
    #         list(map(lambda t: t.data, self.tensor(self.KEYS.TENSOR.BUFFER)))))
    #     logger.debug('UPDATE: {}'.format(
    #         self.tensor(self.KEYS.TENSOR.UPDATE).data))


def _basic_test_by_show_graph():
    from dxl.learn.utils.debug import write_graph
    from dxl.learn.core import GraphInfo
    from srf.graph.master import MasterGraph
    import numpy as np
    name = 'master'
    mg = MasterGraph(np.ones([100] * 3), 2, name, GraphInfo(name, name, False))
    write_graph('/tmp/test_path')
