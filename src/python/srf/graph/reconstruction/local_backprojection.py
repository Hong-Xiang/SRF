import numpy as np
from dxl.learn import Graph
from dxl.learn.tensor import no_op
from dxl.learn.function import dependencies, merge_ops
from tqdm import tqdm


class LocalBackprojectionGraph(Graph):
    class KEYS(Graph.KEYS):
        class TENSOR(Graph.KEYS.TENSOR):
            BACKPROJECTION_STEP = 'backprojection_step'
            UPDATE = 'update'
            X = 'x'
            INIT = 'init'

        class GRAPH(Graph.KEYS.GRAPH):
            MASTER = 'master'
            WORKER = 'worker'

    def __init__(self, name, master_graph, worker_graph):
        super().__init__(name)
        self.graphs[self.KEYS.GRAPH.MASTER] = master_graph
        self.graphs[self.KEYS.GRAPH.WORKER] = worker_graph

    def kernel(self, inputs=None):
        KT = self.KEYS.TENSOR
        m = self.graphs[self.KEYS.GRAPH.MASTER]
        m.make()
        w = self.graphs[self.KEYS.GRAPH.WORKER]
        w.tensors[w.KEYS.TENSOR.X] = self.tensors[self.KEYS.TENSOR.X] = m.tensors[m.KEYS.TENSOR.X]
        w.tensors[w.KEYS.TENSOR.TARGET] = m.tensors[m.KEYS.TENSOR.BUFFER][0]
        w.make()

        self.tensors[KT.INIT] = merge_ops([m.tensors[m.KEYS.TENSOR.INIT],
                                           w.tensors[w.KEYS.TENSOR.INIT]])
        self.tensors[KT.BACKPROJECTION_STEP] = w.tensors[w.KEYS.TENSOR.UPDATE]
        self.tensors[KT.UPDATE] = m.tensors[m.KEYS.TENSOR.UPDATE]

    def run(self, sess=None):
        KT = self.KEYS.TENSOR
        sess.run(self.tensors[KT.INIT])
        sess.run(self.tensors[KT.BACKPROJECTION_STEP])
        sess.run(self.tensors[KT.UPDATE])
        x = sess.run(self.tensors[KT.X].data)
        np.save('bproj.npy', x)
