from dxl.learn.core import Graph, NoOp, ThisSession

from .master import MasterGraph
from .worker import WorkerGraph
import tensorflow as tf
from srf.model.recon_step import ReconStep
# from srf.model.projection import ProjectionToR
# from srf.model.backprojection import BackProjectionToR
from tqdm import tqdm
from dxl.learn.function import ControlDependencies
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
        super().__init__(info,
                         graphs={
                             self.KEYS.GRAPH.MASTER: master_graph,
                             self.KEYS.GRAPH.WORKER: worker_graph
                         },
                         config={
                             self.KEYS.CONFIG.NB_ITERATIONS: nb_iteration,
                         })

    def kernel(self, inputs=None):
        KS, KT = self.KEYS.GRAPH, self.KEYS.TENSOR
        m = self.graphs[KS.MASTER]
        m.make()
        self.tensors[KT.X] = m.tensors[KT.X]
        w = self.graphs[KS.WORKER]
        w.tensors[KT.X] = m.tensors[KT.X]
        w.tensors[w.KEYS.TENSOR.TARGET] = m.tensors[m.KEYS.TENSOR.BUFFER][0]
        w.make()
        with ControlDependencies([m.tensors[m.KEYS.TENSOR.INIT].data,
                                  w.tensors[w.KEYS.TENSOR.INIT].data]):
            self.tensors[KT.INIT] = NoOp()
        self.tensors[KT.RECONSTRUCTION_STEP] = w.tensors[w.KEYS.TENSOR.UPDATE]
        self.tensors[KT.UPDATE] = m.tensors[m.KEYS.TENSOR.UPDATE]

    def run(self, sess=None):
        KT, KC = self.KEYS.TENSOR, self.KEYS.CONFIG
        sess.run(self.tensors[KT.INIT])
        for i in tqdm(range(self.config(KC.NB_ITERATIONS))):
            sess.run(self.tensors[KT.RECONSTRUCTION_STEP])
            sess.run(self.tensors[KT.UPDATE])
            x = sess.run(self.tensors[KT.X])
            np.save('recon_{}.npy'.format(i), x)
