import json
import h5py
import time
import logging
import numpy as np

from tqdm import tqdm

from dxl.learn.core import Barrier, make_distribute_session
from dxl.learn.core import Master, Barrier, ThisHost, ThisSession, Tensor

from .master import MasterGraph
from .worker import WorkerGraphToR
from ...services.utils import print_tensor, debug_tensor
from ...preprocess.preprocess import preprocess as preprocess_tor
from ...preprocess.preprocess import cut_lors
# from ..task.configure import IterativeTaskConfig

# from ...task.data import ImageInfo, LorsInfo, OsemInfo, TorInfo
from .reconstruction_task import ReconstructionTaskBase
from dxl.data.io import load_array

import logging
logging.basicConfig(
    format='[%(levelname)s] %(asctime)s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%a, %d %b %Y %H:%M:%S',
)
logger = logging.getLogger('srf')
# sample_reconstruction_config = {
#     'grid': [150, 150, 150],
#     'center': [0., 0., 0.],
#     'size': [150., 150., 150.],
#     'map_file': './debug/map.npy',
#     'x_lor_files': './debug/xlors.npy',
#     'y_lor_files': './debug/ylors.npy',
#     'z_lor_files': './debug/zlors.npy',
#     'x_lor_shapes': [100, 6],
#     'y_lor_shapes': [200, 6],
#     'z_lor_shapes': [300, 6],
#     'lor_ranges': None,
#     'lor_steps': None,
# }


# tor_config = {
#     'grid': [150, 150, 150],
#     'center': [0., 0., 0.],
#     'size': [150., 150., 150.],
#     'map_file': './debug/map.npy',
#     'lor_file': './debug/lors.npy',
#     'num_iteration': 10,
#     'num_subsets': 10,
#     'lor_ranges': None,
#     'lor_steps': None,
# }


class ToRReconstructionTask(ReconstructionTaskBase):
    worker_graph_cls = WorkerGraphToR

    class KEYS(ReconstructionTaskBase.KEYS):
        class TENSOR(ReconstructionTaskBase.KEYS.TENSOR):
            LORS = WorkerGraphToR.KEYS.TENSOR.LORS
            EFFICIENCY_MAP = WorkerGraphToR.KEYS.TENSOR.EFFICIENCY_MAP

        class CONFIG(ReconstructionTaskBase.KEYS.CONFIG):
            GAUSSIAN_FACTOR = 'gaussian_factor'
            C_FACTOR = 'c_factor'

    @classmethod
    def default_config(self):
        result = super().default_config()
        result.update({
            self.KEYS.CONFIG.GAUSSIAN_FACTOR: 2.35482005,
            self.KEYS.CONFIG.C_FACTOR: 0.15
        })
        return result

    @classmethod
    def parse_task(cls, task_spec):
        result = super().parse_task(task_spec)

        # TODO: move to where these configs were used.
        limit = result['tof']['tof_res'] * result['tor']['c_factor'] / \
            result['tor']['gaussian_factor'] * 3
        result['tof']['tof_sigma2'] = limit * limit / 9
        result['tof']['tof_bin'] = result['tof']['tof_bin'] * \
            result['tor']['c_factor']
        # ts = task_spec
        # ii = ts.image.to_dict()
        return result

    def load_local_data(self, key):
        if key == self.KEYS.TENSOR.LORS:
            c = self.config('lors')
            result = {}
            for a in WorkerGraphToR.AXIS:
                tid = self.config('task_index')
                nb_lors = c['shapes'][a][0] * self.config('nb_subsets')
                spec = {
                    'path_file': c['path_file'],
                    'path_dataset': "{}/{}".format(c['path_dataset'], a),
                    'slices': "[{}:{},:]".format(tid * nb_lors, (tid + 1) * nb_lors)
                }
                result[a] = load_array(spec).astype(np.float32)
            return result
        return super().load_local_data(key)

    def _make_worker_graphs(self):
        KS, KT = self.KEYS.GRAPH, self.KEYS.TENSOR

        if not ThisHost.is_master():
            self.graphs[self.KEYS.GRAPH.WORKER] = [
                None for i in range(self.nb_workers)]
            mg = self.subgraph(KS.MASTER)
            KT = self.KEYS.TENSOR
            MKT = mg.KEYS.TENSOR
            inputs = {
                KT.EFFICIENCY_MAP: self.load_local_data(KT.EFFICIENCY_MAP),
                KT.LORS: self.load_local_data(KT.LORS)
            }

            wg = WorkerGraphToR(mg.tensor(MKT.X), mg.tensor(MKT.BUFFER)[self.task_index], mg.tensor(MKT.SUBSET),
                                inputs=inputs, task_index=self.task_index, name=self.name / 'worker_{}'.format(self.task_index))
            self.graphs[KS.WORKER][self.task_index] = wg
            logger.info("Worker graph {} created.".format(self.task_index))
        else:
            logger.info("Skip make worker graph in master process.")

    def make_steps(self):
        KS = self.KEYS.STEPS
        self._make_init_step(KS.INIT)
        self._make_recon_step(KS.RECON)
        self._make_merge_step(KS.MERGE)
        # assign_step = self._make_assign_step()
        # self.steps = {
        #     KS.INIT: init_step,
        #     KS.RECON: recon_step,
        #     KS.MERGE: merge_step,
        #     # KS.ASSIGN: assign_step
        # }

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
        merge_op = self.master_graph.tensor(
            self.master_graph.KEYS.TENSOR.UPDATE)
        merge_barrier = Barrier(
            name, [self.master_host], self.hosts, [[merge_op]])
        master_op = merge_barrier.barrier(self.master_host)
        worker_ops = [merge_barrier.barrier(h) for h in self.hosts]
        self.add_step(name, master_op, worker_ops)
        return name