from dxl.learn.core import make_distribute_session, barrier_single
from dxl.learn.graph import MasterWorkerTaskBase
from dxl.learn.core import Master, Barrier, ThisHost, ThisSession, Tensor, barrier_single

from dxl.data.io import load_array
from .master import MasterGraph
from .worker import WorkerGraphBase
# from ..graph import MasterGraph
# from ..graph import WorkerGraph
import json
import numpy as np
import logging

logger = logging.getLogger('srf')


class ReconstructionTaskBase(MasterWorkerTaskBase):
    class KEYS(MasterWorkerTaskBase.KEYS):
        class CONFIG(MasterWorkerTaskBase.KEYS.CONFIG):
            NB_SUBSETS = 'nb_subsets'
            EFFICIENCY_MAP = 'efficiency_map'

        class TENSOR(MasterWorkerTaskBase.KEYS.TENSOR):
            X = 'x'
            EFFICIENCY_MAP = 'efficiency_map'
            INIT = 'init'

        class SUBGRAPH(MasterWorkerTaskBase.KEYS.SUBGRAPH):
            pass

    def __init__(self, task_spec, name=None, graph_info=None, *, job=None, task_index=None, cluster_config=None):
        if name is None:
            name = 'distribute_reconstruction_task'
        super().__init__(job=job, task_index=task_index, cluster_config=cluster_config, name=name,
                         graph_info=graph_info, config=self.parse_task(task_spec))
        self.bind_local_data()

    def parse_task(self, task_spec):
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
        result[self.KEYS.CONFIG.EFFICIENCY_MAP] = {
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
        self.subgraphs[self.KEYS.SUBGRAPH.MASTER] = MasterGraph(
            self.load_local_data(self.KEYS.TENSOR.X), name=self.name / 'master')
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
            wg = WorkerGraphBase(mg.tensor(KT.X), mg.tensor(KT.BUFFER)[self.task_index], mg.tensor(KT.SUBSET),
                                 inputs=inputs, task_index=self.task_index, name=self.name / 'worker_{}'.format(self.task_index))
            self.subgraphs[KS.WORKER][self.task_index] = wg

    def bind_local_data(self):
        """
        bind static data into the graph.
        """
        pass

    def _make_barriers(self):
        """
        the
        """
        mg = self.subgraph(self.KEYS.SUBGRAPH.MASTER)
        if ThisHost.is_master():
            self.tensors[self.KEYS.TENSOR.INIT] = barrier_single(
                self.KEYS.TENSOR.INIT, 1 + self.nb_workers, 1 + self.nb_workers,
                mg.tensor(mg.KEYS.TENSOR.INIT)
                self.nb_workers)
        else:
            wg = self.subgraph(self.KEYS.SUBGRAPH.WORKER)[self.task_index]
            self.tensors[self.KEYS.TENSOR.INIT] = barrier_single(
                self.KEYS.TENSOR.INIT, 1 + self.nb_workers, 1 + self.nb_workers,
                wg.tensor(wg.KEYS.TENSOR.INIT),
                self.task_index
            )
