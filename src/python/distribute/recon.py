from dxl.learn.core import DistributeTask, Barrier, make_distribute_session

class DistributeReconTask(DistributeTask):
    class KEYS(DistributeTask.KEYS):
        class STEPS(DistributeTask.KEYS.STEPS):
            INIT = 'init_step'
            RECON = 'recon_step'
            MERGE = 'merge_step'

    def __init__(self, distribute_configs):
        super.__init__(distribute_configs)
        self.steps = {}

    def make_recon_task(self):
        init_step = self._make_init_step()
        recon_step = self._make_recon_step()
        merge_step = self._make_merge_step()
        self.steps = {
            self.KEYS.STEPS.INIT: init_step,
            self.KEYS.RECON: recon_step,
            self.KEYS.MERGE: merge_step,
        }
        make_distribute_session()
        # return init_step, recon_step, merge_step
        
    
    def _make_init_step(self, name='init'):
        init_barrier = Barrier(name, self.hosts, [self.master_host],
                                [[g.tensor(g.KEYS.TENSOR.INIT)]
                                for g in self.worker_graphs])
        master_op = init_barrier.barrier(self.master_host)
        worker_ops = [init_barrier.barrier(h) for h in self.hosts]
        self.add_step(name, master_op, worker_ops)
        return name

    def _make_recon_step(self, name='recon'):
        recons = [[g.tensor(g.KEYS.TENSOR.UPDATE)] for g in self.worker_graphs]
        calculate_barrier = Barrier(
            name, self.hosts, [self.master_host], task_lists=recons)
        master_op = calculate_barrier.barrier(self.master_host)
        worker_ops = [calculate_barrier.barrier(h) for h in self.hosts]
        self.add_step(name, master_op, worker_ops)
        return name

    def _make_merge_step(self, name='merge'):
        """
        """
        merge_op = self.master_graph.tensor(self.master_graph.KEYS.TENSOR.UPDATE)
        merge_barrier = Barrier(name, [self.master_host], self.hosts, [[merge_op]])
        master_op = merge_barrier.barrier(self.master_host)
        worker_ops = [merge_barrier.barrier(h) for h in self.hosts]
        self.add_step(name, master_op, worker_ops)
        return name
