from dxl.learn.core import Graph

class RingEfficiencyMap(Graph):
    class KEYS(Graph.KEYS):
        class TENSOR(Graph.KEYS.TENSOR):
            MAP_STEP = 'map_step'
            INIT = 'init'
        
        class CONFIG(Graph.KEYS.CONFIG):
            TASK_INDEX = 'task_index'
        
        class GRAPH(Graph.KEYS.GRAPH):
            pass
    
    def __init__(self, info, config = None, task_index = None):

        super().__init__(info, config= config)
    
    def kernel(self):
        KS, KT = self.KEYS.GRAPH, self.KEYS.TENSOR

    def run(self, sess):
        
    