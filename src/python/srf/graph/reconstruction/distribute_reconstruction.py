from dxl.learn.graph import MasterWorkerGraphBase
from srf.algorithm import SubgraphMakerBuilder


class MasterWorkerReconstructionGraph(MasterWorkerGraphBase):
    """
    1. Load local data,
    2. Construct MasterGraph and WorkerGraph, link them,
    3. Construct barriers,
    4. convinient run method with init and iterative reconstruction.
    """
    pass
