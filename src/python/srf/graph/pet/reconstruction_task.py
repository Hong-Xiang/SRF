from dxl.learn.core import make_distribute_session, barrier_single
from dxl.learn.graph import MasterWorkerTaskBase
from dxl.learn.core import Master, Barrier, ThisHost, ThisSession, Tensor, barrier_single
from dxl.learn.core.tensor import NoOp

from dxl.data.io import load_array
from .master import MasterGraph
from .worker import WorkerGraphBase
# from ..graph import MasterGraph
# from ..graph import WorkerGraph
import json
import numpy as np
import logging
from tqdm import tqdm

logger = logging.getLogger('srf')


class ReconstructionTaskBase(MasterWorkerTaskBase):
    worker_graph_cls = WorkerGraphBase

    class KEYS(MasterWorkerTaskBase.KEYS):
        class CONFIG(MasterWorkerTaskBase.KEYS.CONFIG):
            NB_SUBSETS = 'nb_subsets'
            EFFICIENCY_MAP = 'efficiency_map'

        class TENSOR(MasterWorkerTaskBase.KEYS.TENSOR):
            X = 'x'
            EFFICIENCY_MAP = 'efficiency_map'
            INIT = 'init'
            RECON = 'recon'
            MERGE = 'merge'

        class SUBGRAPH(MasterWorkerTaskBase.KEYS.SUBGRAPH):
            pass

    def __init__(self, task_spec, name=None, graph_info=None, *, job=None, task_index=None, cluster_config=None):
        if name is None:
            name = 'distribute_reconstruction_task'
        super().__init__(job=job, task_index=task_index, cluster_config=cluster_config, name=name,
                         graph_info=graph_info, config=self.parse_task(task_spec))

    @classmethod
    def parse_task(cls, task_spec):
        """
        Parse TaskSpec to normal dict and update it to config tree.
        """
        # TODO: Add Specs support to CNode/CView, config tree.
        # return {
        #     'task_type': task_spec.task_type,
        #     'work_directory': task_spec.work_directory
        #     'image_info': task_spec.image_info,
        #     'lors': task_spec.lors
        #     'tof': task_spec.tof
        #     'osem': task_spec.osem
        #     'tor': task_spec.tor
        # }
        result = dict(task_spec)
        result[cls.KEYS.CONFIG.EFFICIENCY_MAP] = {
            'path_file': result['image']['map_file']
        }
        del result['image']['map_file']
        return result

    def load_local_data(self, key):
        KC = self.KEYS.CONFIG
        if key == self.KEYS.TENSOR.X:
            shape = self.config('image')['grid']
            return np.ones(shape, dtype=np.float32)
            # return np.ones([128, 128, 104], dtype=np.float32)
        if key == self.KEYS.TENSOR.EFFICIENCY_MAP:
            return load_array(self.config(KC.EFFICIENCY_MAP)).T.astype(np.float32)
        raise KeyError("Known key for load_local_data: {}.".format(key))

    def _make_master_graph(self):
        mg = MasterGraph(
            self.load_local_data(self.KEYS.TENSOR.X), name=self.name / 'master')
        self.subgraphs[self.KEYS.SUBGRAPH.MASTER] = mg
        self.tensors[self.KEYS.TENSOR.MAIN] = mg.tensor(self.KEYS.TENSOR.X)
        self.tensors[self.KEYS.TENSOR.X] = mg.tensor(self.KEYS.TENSOR.X)
        logger.info('Master graph created')

    def _make_worker_graphs(self):
        KS = self.KEYS.SUBGRAPH
        if not ThisHost.is_master():
            self.subgraphs[self.KEYS.SUBGRAPH.WORKER] = [
                None for i in range(self.nb_workers)]
            mg = self.subgraph(KS.MASTER)
            KT = mg.KEYS.TENSOR
            inputs = {
                self.KEYS.TENSOR.EFFICIENCY_MAP: self.load_local_data(
                    self.KEYS.TENSOR.EFFICIENCY_MAP)
            }
            wg = self.worker_graph_cls(mg.tensor(KT.X), mg.tensor(KT.BUFFER)[self.task_index], mg.tensor(KT.SUBSET),
                                       inputs=inputs, task_index=self.task_index, name=self.name / 'worker_{}'.format(self.task_index))
            self.subgraphs[KS.WORKER][self.task_index] = wg
            logger.info("Worker graph {} created.".format(self.task_index))
        else:
            logger.info("Skip make worker graph in master process.")

    def _make_init_barrier(self):
        mg = self.subgraph(self.KEYS.SUBGRAPH.MASTER)
        name = self.name / "barrier_{}".format(self.KEYS.TENSOR.INIT)
        if ThisHost.is_master():
            task = mg.tensor(mg.KEYS.TENSOR.INIT)
            id_join = self.nb_workers
        else:
            wg = self.subgraph(self.KEYS.SUBGRAPH.WORKER)[self.task_index]
            task = wg.tensor(wg.KEYS.TENSOR.INIT)
            id_join = self.task_index
        init_op = barrier_single(name, 1 + self.nb_workers, 1 + self.nb_workers,
                                 task, id_join)
        self.tensors[self.KEYS.TENSOR.INIT] = init_op

    def _make_recon_barrier(self):
        mg = self.subgraph(self.KEYS.SUBGRAPH.MASTER)
        name = self.name / "barrier_{}".format(self.KEYS.TENSOR.RECON)
        if ThisHost.is_master():
            task = None
            id_join = 0
        else:
            wg = self.subgraph(self.KEYS.SUBGRAPH.WORKER)[self.task_index]
            task = wg.tensor(wg.KEYS.TENSOR.UPDATE)
            id_join = None
        recon_op = barrier_single(name, self.nb_workers, 1, task, id_join)
        self.tensors[self.KEYS.TENSOR.RECON] = recon_op

    def _make_merge_barrier(self):
        mg = self.subgraph(self.KEYS.SUBGRAPH.MASTER)
        name = self.name / "barrier_{}".format(self.KEYS.TENSOR.MERGE)
        if ThisHost.is_master():
            task = mg.tensor(mg.KEYS.TENSOR.UPDATE)
            id_join = None
        else:
            wg = self.subgraph(self.KEYS.SUBGRAPH.WORKER)[self.task_index]
            task = None
            id_join = self.task_index
        merge_op = barrier_single(name, 1, self.nb_workers, task, id_join)
        self.tensors[self.KEYS.TENSOR.MERGE] = merge_op

    def _make_barriers(self):
        """
        the
        """
        self._make_init_barrier()
        self._make_recon_barrier()
        self._make_merge_barrier()

    def _run_with_info(self, key):
        logger.info('Start {}...'.format(key))
        ThisSession.run(self.tensor(key))
        logger.info('{} Complete.'.format(key))

    def run_task(self):
        KT = self.KEYS.TENSOR
        KC = self.KEYS.CONFIG
        logger.info(
            'Reconstruction Task::job:{}/task_index:{}'.format(self.job, self.task_index))
        self._run_with_info(KT.INIT)

        nb_iterations = self.config('osem')['nb_iterations']
        nb_subsets = self.config('osem')['nb_subsets']
        image_name = self.config('image')['name']
        for i in tqdm(range(nb_iterations), ascii=True):
            for j in tqdm(range(nb_subsets), ascii=True):
                self._run_with_info(KT.RECON)
                self._run_with_info(KT.MERGE)
                self._print_x()
                self._save_result('{}_{}_{}.npy'.format(image_name, i, j))

    @MasterWorkerTaskBase.master_only
    def _save_result(self, path):
        x = ThisSession.run(self.tensor(self.KEYS.TENSOR.X))
        np.save(path, x)

    def _print_x(self):
        x = ThisSession.run(self.tensor(self.KEYS.TENSOR.X))
        print('X value:', x, sep='\n')
