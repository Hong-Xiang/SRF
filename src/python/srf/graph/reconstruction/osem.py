import tensorflow as tf

from dxl.learn.core import NoOp, Variable
from dxl.learn.graph import MasterWorkerTaskBase

from .master import MasterGraph


class OSEMMasterGraph(MasterGraph):
    class KEYS(MasterGraph.KEYS):
        class TENSOR(MasterGraph.KEYS.TENSOR):
            SUBSET = 'subset'
            INC_SUBSET = 'inc_subset'

        class CONFIG(MasterGraph.KEYS.CONFIG):
            NB_SUBSETS = 'nb_subsets'

    def __init__(self, info, config=None, *, initial_image, nb_workers=None, nb_subsets=None):
        config = self._update_config_if_not_none(config, {
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
        with tf.control_dependencies([self.tensor(KT.X).init().data, self.tensor(KT.SUBSET).init().data]):
            self.tensors[self.KEYS.TENSOR.INIT] = NoOp()

    def _bind_increase_subset(self):
        KT = self.KEYS.TENSOR
        with tf.control_dependencies([self.tensor(KT.UPDATE).data, self.tensor(KT.INC_SUBSET).data]):
            self.tensors[KT.UPDATE] = NoOp()


class OSEMReconstructionGraph(MasterWorkerTaskBase):
    def __init__(self, info, config=None, tensors=None, subgraphs=None):

        pass

    def _make_master_graph(self):
        return OSEMMasterGraph()

    def _make_worker_graph(self, task_index):
        pass
