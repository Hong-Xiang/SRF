from dxl.learn.core import Graph


class LocalReconstructionGraph(Graph):
    def __init__(self, info, master_data_loader, worker_data_loader, *, config):
        self._master_data_loader = master_data_loader
        self._worker_data_loader = worker_data_loader
        super().__init__(info, config=config)

    def kernel(self):
        pass

    def run(self):
        pass
