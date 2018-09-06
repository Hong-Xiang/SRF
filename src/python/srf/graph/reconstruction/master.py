import tensorflow as tf
from doufo.tensor import sum_
from dxl.learn.tensor import no_op, variable_from_tensor, variable, initializer, sum_
from dxl.learn import Graph
from dxl.learn.function import dependencies, merge_ops

from srf.utils import logger


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

    def __init__(self, name, loader=None, nb_workers=None, is_renormalization=False):
        super().__init__(name)
        self.config.update(self.KEYS.CONFIG.NB_WORKERS, nb_workers)
        self.config.update_value_with_default(self.KEYS.CONFIG.RENORMALIZATION, is_renormalization, False)
        self._loader = loader

    @logger.after.debug('Master graph constructed.')
    def kernel(self, inputs=None):
        self._construct_x()
        self._construct_init()
        self._construct_summation()

    @property
    def nb_workers(self):
        return self.config(self.KEYS.CONFIG.NB_WORKERS)

    def _construct_x(self):
        x = self.tensors[self.KEYS.TENSOR.X] = variable_from_tensor(self._loader.load(self), self.KEYS.TENSOR.X)
        self.tensors[self.KEYS.TENSOR.BUFFER] = [
            variable(shape=x.shape,
                     dtype=x.dtypem,
                     name=f'{self.KEYS.TENSOR.BUFFER}_{i}')
            for i in range(self.config[self.KEYS.CONFIG.NB_WORKERS])
        ]

    def _construct_init(self):
        to_init = [self.tensors[self.KEYS.TENSOR.X],
                   self.tensors[self.KEYS.TENSOR.BUFFER]]
        self.tensors[self.KEYS.TENSOR.INIT] = merge_ops([initializer(t) for t in to_init])

    def _construct_summation(self):
        KT = self.KEYS.TENSOR
        x_s = sum_(self.tensors[KT.BUFFER], axis=0)
        if self.config[self.KEYS.CONFIG.RENORMALIZATION]:
            x_s = x_s / sum_(x_s) * sum_(self.tensors[KT.X].data)
        self.tensors[KT.UPDATE] = self.tensors[KT.X].assign(x_s)


class MasterGraphWithGlobalStep(MasterGraph):
    def _construct_summation(self):
        super()._construct_summation()
        gs = tf.train.get_or_create_global_step()
        gsa = gs.assign(gs + 1)
        with dependencies([self.tensors[self.KEYS.TENSOR.UPDATE], gsa]):
            self.tensors[self.KEYS.TENSOR.UPDATE] = no_op()


class OSEMMasterGraph(MasterGraph):
    class KEYS(MasterGraph.KEYS):
        class TENSOR(MasterGraph.KEYS.TENSOR):
            SUBSET = 'subset'
            INC_SUBSET = 'inc_subset'

        class CONFIG(MasterGraph.KEYS.CONFIG):
            NB_SUBSETS = 'nb_subsets'

    def __init__(self, info, *, loader=None, nb_workers=None, nb_subsets=None, is_renormalization=None):
        super().__init__(info,
                         loader=loader, nb_workers=nb_workers, is_renormalization=is_renormalization)
        self.config.update({self.KEYS.CONFIG.NB_SUBSETS: nb_subsets})

    @logger.after.debug('Master graph constructed.')
    def kernel(self, inputs=None):
        self._construct_x()
        self._construct_subset()
        self._construct_init()
        self._construct_summation()
        self._bind_increase_subset()

    @property
    def nb_subsets(self):
        return self.config(self.KEYS.CONFIG.NB_SUBSETS)

    def _construct_subset(self):
        subset = variable_from_tensor(0, self.KEYS.TENSOR.SUBSET)
        self.tensors[self.KEYS.TENSOR.SUBSET] = subset
        with tf.name_scope(self.KEYS.TENSOR.INC_SUBSET):
            self.tensors[self.KEYS.TENSOR.INC_SUBSET] = subset.assign(
                (subset.data + 1) % self.config(self.KEYS.CONFIG.NB_SUBSETS))

    def _construct_init(self):
        KT = self.KEYS.TENSOR
        super()._construct_init()
        with dependencies([self.tensors(KT.INIT), self.tensors[KT.SUBSET].init()]):
            self.tensors[self.KEYS.TENSOR.INIT] = no_op()

    def _bind_increase_subset(self):
        KT = self.KEYS.TENSOR
        with dependencies([self.tensors[KT.UPDATE], self.tensors[KT.INC_SUBSET]]):
            self.tensors[KT.UPDATE] = no_op()
