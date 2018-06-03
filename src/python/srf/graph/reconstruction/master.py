from typing import Iterable

import numpy as np
import tensorflow as tf

from dxl.learn.core import Graph, Tensor, NoOp
from dxl.learn.distribute import DistributeGraphInfo, Master
from dxl.learn.core.tensor import Variable
from srf.utils import logger
from dxl.learn.model import Summation

# from .utils import constant_tensor, variable_tensor


class MasterGraph(Graph):
    class KEYS(Graph.KEYS):
        class CONFIG(Graph.KEYS.CONFIG):
            NB_WORKERS = 'nb_workers'
            RENORMALIZATION = 'renormalization'

        class TENSOR(Graph.KEYS.TENSOR):
            X = 'x'
            BUFFER = 'x_buffer'
            UPDATE = 'x_update'
            INIT = 'init'

        class SUBGRAPH(Graph.KEYS.SUBGRAPH):
            SUMMATION = 'summation'

    def __init__(self, info, *, config=None, initial_image, nb_workers=None):
        """
        `initial_image`: numpy.ndarray, initial image ndarray.
        """
        config = self._parse_input_config(config, {
            self.KEYS.CONFIG.NB_WORKERS: nb_workers,
            self.KEYS.CONFIG.RENORMALIZATION: False,
        })
        self._initial_image = initial_image
        super().__init__(info, config=config)

    @logger.after.debug('Master graph constructed.')
    def kernel(self):
        self._construct_x()
        # self._construct_subset()
        self._construct_init()
        self._construct_summation()
        # self._debug_info()

    @property
    def nb_workers(self):
        return self.config(self.KEYS.CONFIG.NB_WORKERS)

    def _construct_x(self):
        x = Variable(self.info.child_tensor(self.KEYS.TENSOR.X),
                     initializer=self._initial_image)
        self._initial_image = None
        buffer = [
            Variable(
                self.info.child_tensor(
                    '{}_{}'.format(self.KEYS.TENSOR.BUFFER, i)),
                shape=x.shape,
                dtype=x.dtype) for i in range(self.config(self.KEYS.CONFIG.NB_WORKERS))
        ]
        self.tensors[self.KEYS.TENSOR.X] = x
        self.tensors[self.KEYS.TENSOR.BUFFER] = buffer

    def _construct_init(self):
        KT = self.KEYS.TENSOR
        with tf.control_dependencies([self.tensor(KT.X).init().data]):
            self.tensors[self.KEYS.TENSOR.INIT] = NoOp()

    def _construct_summation(self):
        gs = tf.train.get_or_create_global_step()
        gsa = gs.assign(gs + 1)
        KT, KG = self.KEYS.TENSOR, self.KEYS.SUBGRAPH
        summation = self.subgraph(KG.SUMMATION, lambda g, n: Summation(
            g.info.child_scope(n), self.tensor(KT.BUFFER)))
        x_s = summation()
        if self.config(self.KEYS.CONFIG.RENORMALIZATION):
            sum_s = tf.reduce_sum(x_s.data)
            sum_x = tf.reduce_sum(self.tensor(TK.X).data)
            x_s = x_s.data / sum_s * sum_x
        x_u = self.tensor(KT.X).assign(x_s)
        with tf.control_dependencies([x_u.data, gsa]):
            self.tensors[KT.UPDATE] = NoOp()
        return x_u

    # @property
    # def x(self):
    #     return self.tensor(self.KEYS.TENSOR.X)
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
