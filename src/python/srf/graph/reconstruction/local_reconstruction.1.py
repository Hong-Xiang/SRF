from dxl.learn.core import Graph, NoOp, SubgraphMakerTable

from .master import MasterGraph
from .worker import WorkerGraph
import tensorflow as tf
from srf.model.recon_step import ReconStep
from srf.model._projection import ProjectionToR
from srf.model._backprojection import BackProjectionToR
from tqdm import tqdm
import numpy as np


class LocalReconstructionGraph(Graph):
    class KEYS(Graph.KEYS):
        class TENSOR(Graph.KEYS.TENSOR):
            RECONSTRUCTION_STEP = 'reconstruction_step'
            UPDATE = 'update'
            X = 'x'
            INIT = 'init'

        class CONFIG(Graph.KEYS.CONFIG):
            NB_ITERATIONS = 'nb_iterations'

        class GRAPH(Graph.KEYS.GRAPH):
            MASTER = 'master'
            WORKER = 'worker'

    def __init__(self, info, master_graph, worker_graph, *, nb_iteration=10):
        super().__init__(info, config={
            self.KEYS.CONFIG.NB_ITERATIONS: nb_iteration,
        })

    def kernel(self):
        KS, KT = self.KEYS.GRAPH, self.KEYS.TENSOR
        # m = self.graphs[KS.MASTER] = MasterGraph(self.info.child_scope(KS.MASTER),
        #                                             loader=self._master_data_loader, nb_workers=1)
        # self.tensors[KT.X] = m.tensor(KT.X)
        # w = self.graphs[KS.WORKER] = WorkerGraph(self.info.child_scope(KS.WORKER), m.tensor(
        #     KT.X), m.tensor(m.KEYS.TENSOR.BUFFER)[0], loader=self._worker_data_loader, recon_step_cls=ReconStep)
        # with tf.control_dependencies([m.tensor(m.KEYS.TENSOR.INIT).data, w.tensor(w.KEYS.TENSOR.INIT).data]):
        #     self.tensors[KT.INIT] = NoOp()
        # self.tensors[KT.RECONSTRUCTION_STEP] = w.tensor(w.KEYS.TENSOR.UPDATE)
        # self.tensors[KT.UPDATE] = m.tensor(m.KEYS.TENSOR.UPDATE)
        self.graphs[KS.MASTER].make() = MasterGraph(self.info.child_scope(KS.MASTER),
                                                    loader=self._master_data_loader, nb_workers=1)
        self.tensors[KT.X] = m.get_or_create_tensor(KT.X)
        w = self.graphs[KS.WORKER] = WorkerGraph(self.info.child_scope(KS.WORKER), m.get_or_create_tensor(
            KT.X), m.get_or_create_tensor(m.KEYS.TENSOR.BUFFER)[0], loader=self._worker_data_loader, recon_step_cls=ReconStep)
        with tf.control_dependencies([m.get_or_create_tensor(m.KEYS.TENSOR.INIT).data, w.get_or_create_tensor(w.KEYS.TENSOR.INIT).data]):
            self.tensors[KT.INIT] = NoOp()
        self.tensors[KT.RECONSTRUCTION_STEP] = w.get_or_create_tensor(
            w.KEYS.TENSOR.UPDATE)
        self.tensors[KT.UPDATE] = m.get_or_create_tensor(m.KEYS.TENSOR.UPDATE)

    def run(self, sess):
        KT, KC = self.KEYS.TENSOR, self.KEYS.CONFIG
        # sess.run(self.tensor(KT.INIT))
        # for i in tqdm(range(self.config(KC.NB_ITERATIONS))):
        #     sess.run(self.tensor(KT.RECONSTRUCTION_STEP))
        #     sess.run(self.tensor(KT.UPDATE))
        #     x = sess.run(self.tensor(KT.X))
        #     np.save('recon_{}.npy'.format(i), x)
        sess.run(self.get_or_create_tensor(KT.INIT))
        for i in tqdm(range(self.config(KC.NB_ITERATIONS))):
            sess.run(self.get_or_create_tensor(KT.RECONSTRUCTION_STEP))
            sess.run(self.get_or_create_tensor(KT.UPDATE))
            x = sess.run(self.get_or_create_tensor(KT.X))
            np.save('recon_{}.npy'.format(i), x)
