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
            IS_INC_GLOBAL_STEP = 'is_inc_global_step'

        class TENSOR(Graph.KEYS.TENSOR):
            X = 'x'
            BUFFER = 'x_buffer'
            UPDATE = 'x_update'
            INIT = 'init'

        class SUBGRAPH(Graph.KEYS.SUBGRAPH):
            SUMMATION = 'summation'

    def __init__(self, info, *, config=None, initial_image=None, nb_workers=None):
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
        self._construct_init()
        self._construct_summation()

    @property
    def nb_workers(self):
        return self.config(self.KEYS.CONFIG.NB_WORKERS)

    def _construct_x(self):
        KT, KC = self.KEYS.TENSOR, self.KEYS.CONFIG
        x = self.tensors[KT.X] = Variable(self.info.child_tensor(KT.X),
                                          initializer=self._initial_image)
        self.tensors[KT.BUFFER] = [
            Variable(
                self.info.child_tensor(
                    '{}_{}'.format(KT.BUFFER, i)),
                shape=x.shape,
                dtype=x.dtype) for i in range(self.config(KC.NB_WORKERS))
        ]

    def _construct_init(self):
        KT = self.KEYS.TENSOR
        to_init = [self.tensor(KT.X)] + self.tensor(KT.BUFFER)
        with tf.control_dependencies([t.init().data for t in to_init]):
            self.tensors[KT.INIT] = NoOp()

    def _construct_summation(self):
        KT, KS = self.KEYS.TENSOR, self.KEYS.SUBGRAPH
        summation = self.subgraphs[KS.SUMMATION] = Summation(
            self.info.child_scope(KS.SUMMATION), self.tensor(KT.BUFFER))
        x_s = summation()
        if self.config(self.KEYS.CONFIG.RENORMALIZATION):
            sum_s = tf.reduce_sum(x_s.data)
            sum_x = tf.reduce_sum(self.tensor(TK.X).data)
            x_s = x_s.data / sum_s * sum_x
        self.tensors[KT.UPDATE] = self.tensor(KT.X).assign(x_s)


class MasterGraphWithGlobalStep(MasterGraph):
    def _construct_summation(self):
        super()._construct_summation()
        gs = tf.train.get_or_create_global_step()
        gsa = gs.assign(gs + 1)
        with tf.control_dependencies([self.tensor(KT.UPDATE).data, gsa]):
            self.tensors[KT.UPDATE] = NoOp()


class OSEMMasterGraph(MasterGraph):
    class KEYS(MasterGraph.KEYS):
        class TENSOR(MasterGraph.KEYS.TENSOR):
            SUBSET = 'subset'
            INC_SUBSET = 'inc_subset'

        class CONFIG(MasterGraph.KEYS.CONFIG):
            NB_SUBSETS = 'nb_subsets'

    def __init__(self, info, config=None, *, initial_image, nb_workers=None, nb_subsets=None):
        config = self._parse_input_config(config, {
            self.KEYS.CONFIG.NB_SUBSETS: nb_subsets})
        super().__init__(info, config=config, initial_image=initial_image, nb_workers=nb_workers)

    def kernel(self):
        self._construct_x()
        self._construct_subset()
        self._construct_init()
        self._construct_summation()
        self._bind_increase_subset()

    @property
    def nb_subsets(self):
        return self.config(self.KEYS.CONFIG.NB_SUBSETS)

    def _construct_subset(self):
        subset = Variable(self.info.child_tensor(
            self.KEYS.TENSOR.SUBSET), initializer=0)
        self.tensors[self.KEYS.TENSOR.SUBSET] = subset
        with tf.name_scope(self.KEYS.TENSOR.INC_SUBSET):
            self.tensors[self.KEYS.TENSOR.INC_SUBSET] = subset.assign(
                (subset.data + 1) % self.config(self.KEYS.CONFIG.NB_SUBSETS))

    def _construct_init(self):
        KT = self.KEYS.TENSOR
        super()._construct_init()
        with tf.control_dependencies([self.tensor(KT.INIT).data, self.tensor(KT.SUBSET).init().data]):
            self.tensors[self.KEYS.TENSOR.INIT] = NoOp()

    def _bind_increase_subset(self):
        KT = self.KEYS.TENSOR
        with tf.control_dependencies([self.tensor(KT.UPDATE).data, self.tensor(KT.INC_SUBSET).data]):
            self.tensors[KT.UPDATE] = NoOp()

#     # @property
#     # def x(self):
#     #     return self.tensor(self.KEYS.TENSOR.X)
#         # def _debug_info(self):
#         #     logger.debug('Master graph constructed.')
#         #     logger.debug('X: {}'.format(self.tensor(self.KEYS.TENSOR.X).data))
#         #     logger.debug('BUFFER: {}'.format(
#         #         list(map(lambda t: t.data, self.tensor(self.KEYS.TENSOR.BUFFER)))))
#         #     logger.debug('UPDATE: {}'.format(
#         #         self.tensor(self.KEYS.TENSOR.UPDATE).data))


# def _basic_test_by_show_graph():
#     from dxl.learn.utils.debug import write_graph
#     from dxl.learn.core import GraphInfo
#     from srf.graph.master import MasterGraph
#     import numpy as np
#     name = 'master'
#     mg = MasterGraph(np.ones([100] * 3), 2, name, GraphInfo(name, name, False))
#     write_graph('/tmp/test_path')
