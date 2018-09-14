import numpy as np

from dxl.learn import Graph
from dxl.learn.tensor import dependencies
from dxl.learn.tensor import no_op


class LocalEfficiencyMapGraph(Graph):
    class KEYS(Graph.KEYS):
        class TENSOR(Graph.KEYS.TENSOR):
            MAP_STEP = 'map_step'
            UPDATE = 'update'
            X = 'x'
            INIT = 'init'

        class TASK:
            INIT = 'init'
            MAP_STEP = 'map_step'
            RUN = 'run'

        class GRAPH(Graph.KEYS.GRAPH):
            MASTER = 'master'
            WORKER = 'worker'

    def __init__(self, name, master, worker):
        super().__init__(name)
        self.graphs[self.KEYS.GRAPH.MASTER] = master
        self.graphs[self.KEYS.GRAPH.WORKER] = worker

    def build(self):
        KT = self.KEYS.TENSOR
        m = self.graphs[self.KEYS.GRAPH.MASTER]
        m.make()
        w = self.graphs[self.KEYS.GRAPH.WORKER]
        w.tensors[w.KEYS.TENSOR.X] = self.tensors[self.KEYS.TENSOR.X] = m.tensors[m.KEYS.TENSOR.X]
        w.tensors[w.KT.TARGET] = m.tensors[m.KEYS.TENSOR.BUFFER][0]
        w.make()

        with dependencies([m.tensors[m.KEYS.TENSOR.INIT],
                           w.tensors[w.KEYS.TENSOR.INIT]]):
            self.tensors[KT.INIT] = no_op()

        self.tensors[KT.MAP_STEP] = w.tensors[w.KEYS.TENSOR.RESULT]
        self.tensors[KT.UPDATE] = m.tensors[m.KEYS.TENSOR.UPDATE]

    def init(self, session):
        return session.run(self.tensors[self.KEYS.TENSOR.INIT])

    def calculate(self, session):
        return session.run(self.tensors[self.KEYS.UPDATE])

    def fetch(self, session):
        return session.run(self.tensors(self.KEYS.TENSOR.X))

    def run(self, session):
        self.init(session)
        self.calculate_efficiency_map(session)
        return self.fetch(session)
        np.save('effmap.npy', x)
