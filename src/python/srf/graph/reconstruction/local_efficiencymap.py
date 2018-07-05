import tensorflow as tf
import numpy as np

from dxl.learn.core import Graph, NoOp
from .effmap_worker import WorkerGraph
from .effmap_master import MasterGraph

from srf.model.map_step import MapStep
from srf.model.backprojection import BackProjectionTOR
from srf.physics.tor_map import ToRMapModel


class LocalEfficiencyMapGraph(Graph):
    class KEYS(Graph.KEYS):
        class TENSOR(Graph.KEYS.TENSOR):
            MAP_STEP = 'map_step'
            UPDATE = 'update'
            X = 'x'
            INIT = 'init'

        class GRAPH(Graph.KEYS.GRAPH):
            MASTER = 'master'
            WORKER = 'worker'

    def __init__(self, info,
                 *,
                 master,
                 worker,
                 config=None):
        graphs = {}
        graphs.update({
            self.KEYS.GRAPH.MASTER: master,
            self.KEYS.GRAPH.WORKER: worker
        })
        super().__init__(info, config=config, graphs=graphs)

    def kernel(self):
        KS, KT = self.KEYS.GRAPH, self.KEYS.TENSOR
        m = self.graphs[KS.MASTER]

        w = self.graphs[KS.WORKER]

        self.tensors[KT.X] = m.tensors[KT.X]
        w.tensors[KT.X] = self.tensors[KT.X]
        w.tensors[w.KT.TARGET] = m.tensors[m.KEYS.TENSOR.BUFFER][0]

        with tf.control_dependencies([m.get_or_create_tensor(m.KEYS.TENSOR.INIT).data, w.get_or_create_tensor(w.KEYS.TENSOR.INIT).data]):
            self.tensors[KT.INIT] = NoOp()
        
        self.tensors[KT.MAP_STEP] = w.tensors[w.KEYS.TENSOR.RESULT] 
        self.tensors[KT.UPDATE] = m.tensors[m.KEYS.TENSOR.UPDATE]
    
    def run(self, sess):
        KT, KC = self.KEYS.TENSOR, self.KEYS.CONFIG
        sess.run(self.tensors[KT.INIT])
        sess.run(self.tensors[KT.UPDATE])
        x = sess.run(self.get_or_create_tensor(KT.X))
        np.save('effmap.npy', x)
