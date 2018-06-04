import json
import h5py
import time
import logging
import numpy as np

from tqdm import tqdm

from dxl.learn.core import Barrier, make_distribute_session
from dxl.learn.core import Master, Barrier, ThisHost, ThisSession, Tensor
from dxl.data.io import load_array

from ...services.utils import print_tensor, debug_tensor
from ...preprocess.preprocess import preprocess as preprocess_tor
from ...preprocess.preprocess import cut_lors

from .master_osem import OsemMasterGraph
from .worker_app_tor import TorWorkerGraph

from .task_base_osem import OsemTaskBase


logging.basicConfig(
    format='[%(levelname)s] %(asctime)s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%a, %d %b %Y %H:%M:%S',
)
logger = logging.getLogger('srf')

class ToRReconstructionTask(OsemTaskBase):
    master_graph_cls = OsemMasterGraph
    worker_graph_cls = TorWorkerGraph

    class KEYS(OsemTaskBase.KEYS):
        class TENSOR(OsemTaskBase.KEYS.TENSOR):
            LORS = TorWorkerGraph.KEYS.TENSOR.LORS
            EFFICIENCY_MAP = TorWorkerGraph.KEYS.TENSOR.EFFICIENCY_MAP

        class CONFIG(OsemTaskBase.KEYS.CONFIG):
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
        KS, KT = self.KEYS.SUBGRAPH, self.KEYS.TENSOR

        if not ThisHost.is_master():
            self.subgraphs[self.KEYS.SUBGRAPH.WORKER] = [
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
            self.subgraphs[KS.WORKER][self.task_index] = wg
            logger.info("Worker graph {} created.".format(self.task_index))
        else:
            logger.info("Skip make worker graph in master process.")
