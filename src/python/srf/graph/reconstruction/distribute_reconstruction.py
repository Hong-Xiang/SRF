from dxl.learn.graph import MasterWorkerTaskBase


class MasterWorkerReconstructionGraph(MasterWorkerTaskBase):
    """
    1. Load local data,
    2. Construct MasterGraph and WorkerGraph, link them,
    3. Construct barriers,
    4. convinient run method with init and iterative reconstruction.
    """

    def __init__(self, info, master_local_loader_cls, worker_local_loader_cls, *, job, task_index, cluster, config, graphs):
        super().__init__()