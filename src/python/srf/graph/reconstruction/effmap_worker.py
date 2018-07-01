from dxl.learn.core import Graph, Tensor, Variable, Constant, NoOp

class WorkerGraph(Graph):
    """Base class of a worker graph to compute efficiency map. 

    """
    class KEYS(Graph.KEYS):
        class CONFIG(Graph.KEYS.CONFIG):
            TASK_INDEX = 'task_index'
        
        class TENSOR(Graph.KEYS.TENSOR):
            X = 'x'
            TARGET = 'target'
            INIT = 'init'

        class SUBGRAPH(Graph.KEYS.SUBGRAPH):
            EFFMAP = 'effmap'
        

    def __init__(self,
                 info, 
                 x: Tensor,
                 x_target: Variable,
                 *
                 loader = None,
                 tensors = None,
                 graphs=None,
                 config = None):
        self._loader = loader
        if tensors is None:
            tensors = {}