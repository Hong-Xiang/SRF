import json
import h5py
import time
import logging
import numpy as np
from scipy import sparse
import scipy.io as sio

from tqdm import tqdm

from dxl.learn.core import Barrier, make_distribute_session
from dxl.learn.core import Master, Barrier, ThisHost, ThisSession, Tensor

from .master_sino import MasterGraph
from .worker_sino import WorkerGraphSINO
from ...services.utils import print_tensor, debug_tensor
#from ...preprocess.preprocess import preprocess as preprocess_tor

from ...task.sinodatanew import ImageInfo, SinoInfo, ReconInfo, MatrixInfo
from .recon_task_new import ReconstructionTaskBase
from dxl.data.io import load_array
from ...preprocess import preprocess_sino

import logging
logging.basicConfig(
    format='[%(levelname)s] %(asctime)s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%a, %d %b %Y %H:%M:%S',
)
logger = logging.getLogger('sino')



class SinoReconstructionTask(ReconstructionTaskBase):
    worker_graph_cls = WorkerGraphSINO

    class KEYS(ReconstructionTaskBase.KEYS):
        class TENSOR(ReconstructionTaskBase.KEYS.TENSOR):
            SINO = WorkerGraphSINO.KEYS.TENSOR.SINOS
            EFFICIENCY_MAP = WorkerGraphSINO.KEYS.TENSOR.EFFICIENCY_MAP
            MATRIX = WorkerGraphSINO.KEYS.TENSOR.MATRIX

        # class CONFIG(ReconstructionTaskBase.KEYS.CONFIG):
        #     GAUSSIAN_FACTOR = 'gaussian_factor'
        #     C_FACTOR = 'c_factor'

    # @classmethod
    # def default_config(self):
    #     result = super().default_config()
    #     result.update({
    #         self.KEYS.CONFIG.GAUSSIAN_FACTOR: 2.35482005,
    #         self.KEYS.CONFIG.C_FACTOR: 0.15
    #     })
    #     return result

    @classmethod
    def parse_task(cls, task_spec):
        result = super().parse_task(task_spec)

        # TODO: move to where these configs were used.
        # limit = result['tof']['tof_res'] * result['tor']['c_factor'] / \
        #     result['tor']['gaussian_factor'] * 3
        # result['tof']['tof_sigma2'] = limit * limit / 9
        # result['tof']['tof_bin'] = result['tof']['tof_bin'] * \
        #     result['tor']['c_factor']
        ts = task_spec
        ii = ts.image
        return result
        self.image_info = ImageInfo(ii['grid'],
                                    ii['name'],
                                    ii['map_file'])

        oi = ts.recon.to_dict()
        self.osem_info = ReconInfo(oi['nb_iterations'],
                                  oi['save_interval'])

        self.sino_file = ts.sino.path_file
        self.matrix_file = ts.matrix.path_file
        
        self.sino_info = SinoInfo(
            self.sino_file,
            ts.sino.shape,
            ts.sino.step,
            None
        )

        self.matrix_info = MatrixInfo(
            self.matrix_file,
            ts.matrix.shape,
            ts.matrix.step,
            None)


    def load_local_data(self, key):
        if key == self.KEYS.TENSOR.SINO:
            c = self.config('sino')
            result = {}
            tid = self.config('task_index')
            nb_sino = c['shapes'][0]
            spec = {
                'path_file': c['path_file'],
                'path_dataset': "{}".format(c['path_dataset']),
                'slices': "[{}:{},:]".format(tid * nb_sino, (tid + 1) * nb_sino)
            }
            result = load_array(spec).astype(np.float32)
            # with h5py.File(c['path_file']) as fin:
            #     data = fin[c['path_dataset']][:]
            # datanew = np.load('/home/twj2417/SRF/SRF/config/jaszczak.npy')
            # data = preprocess_sino.preprocess_sino(datanew)
            # data = np.load('/home/twj2417/SRF/SRF/config/jaszczaknewnew.npy')
            # result = data[(tid*nb_sino):((tid+1)*nb_sino),:]
            return result
        if key == self.KEYS.TENSOR.MATRIX:
            c = self.config('matrix')
            result = {}
            tid = self.config('task_index')
            nb_matrix = c['shapes'][0]
            #slices = [(tid*nb_matrix):((tid+1)*nb_matrix),:]
            # spec = {
            #     'path_file': c['path_file'],
            #     'path_dataset': "{}".format(c['path_dataset']),
            #     'slices': "[{}:{},:]".format(tid * nb_matrix, (tid + 1) * nb_matrix)
            # }
            dataset = sio.loadmat(c['path_file'])
            data = dataset[c['path_dataset']]
            data = sparse.csc_matrix(data)
            # slices=spec['slices']
            result = data[(tid*nb_matrix):((tid+1)*nb_matrix),:]
            result = sparse.coo_matrix(result)
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
                KT.SINO: self.load_local_data(KT.SINO),
                KT.MATRIX: self.load_local_data(KT.MATRIX)
            }

            wg = WorkerGraphSINO(mg.tensor(MKT.X), mg.tensor(MKT.BUFFER)[self.task_index], 
                                inputs=inputs, task_index=self.task_index, name=self.name / 'worker_{}'.format(self.task_index))
            self.subgraphs[KS.WORKER][self.task_index] = wg
            logger.info("Worker graph {} created.".format(self.task_index))
        else:
            logger.info("Skip make worker graph in master process.")


    def make_steps(self):
        KS = self.KEYS.STEPS
        self._make_init_step(KS.INIT)
        self._make_recon_step(KS.RECON)
        self._make_merge_step(KS.MERGE)

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

