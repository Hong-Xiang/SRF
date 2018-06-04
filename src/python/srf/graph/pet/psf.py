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

from .master import MasterGraph
from .worker_psf import WorkerGraphPSF
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


class PSFReconstructionTask(ReconstructionTaskBase):
    worker_graph_cls = WorkerGraphPSF

    class KEYS(ReconstructionTaskBase.KEYS):
        class TENSOR(ReconstructionTaskBase.KEYS.TENSOR):
            LORS = WorkerGraphPSF.KEYS.TENSOR.LORS
            EFFICIENCY_MAP = WorkerGraphPSF.KEYS.TENSOR.EFFICIENCY_MAP
            PSF_MATRIX = WorkerGraphPSF.KEYS.TENSOR.PSF_MATRIX

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
        self.image_info = ImageInfo(ii['grid'],
                                    ii['center'],
                                    ii['size'],
                                    ii['name'],
                                    ii['map_file'])
        self.kernel_width = ts.tor.kernel_width

        oi = ts.osem.to_dict()
        self.osem_info = OsemInfo(oi['nb_iterations'],
                                  oi['nb_subsets'],
                                  oi['save_interval'])

        self.lors_file = ts.lors.path_file
        tofi = ts.tof.to_dict()
        self.tof_info = TorInfo(tofi['tof_res'],
                                tofi['tof_bin'])
        XYZ = ['x', 'y', 'z']
        self.lors_info = LorsInfo(
            {a: self.lors_file for a in XYZ},
            {a: ts.lors.shape[a] for a in XYZ},
            {a: ts.lors.step[a] for a in XYZ},
            None
        )
        # self.lor_info = LorInfo(
        #     {a: ti['{}_lor_files'.format(a)]
        #      for a in ['x', 'y', 'z']},
        #     {a: ti['{}_lor_shapes'.format(a)]
        #      for a in ['x', 'y', 'z']}, ti['lor_ranges'], ti['lor_steps'])
        limit = ts.tof.tof_res * ts.tor.c_factor / ts.tor.gaussian_factor * 3
        self.tof_sigma2 = limit * limit / 9
        self.tof_bin = self.tof_info.tof_bin * self.c_factor

    def load_local_data(self, key):
        if key == self.KEYS.TENSOR.LORS:
            c = self.config('lors')
            result = {}
            for a in WorkerGraphPSF.AXIS:
                tid = self.config('task_index')
                nb_lors = c['shapes'][a][0] * self.config('nb_subsets')
                spec = {
                    'path_file': c['path_file'],
                    'path_dataset': "{}/{}".format(c['path_dataset'], a),
                    'slices': "[{}:{},:]".format(tid * nb_lors, (tid + 1) * nb_lors)
                }
                result[a] = load_array(spec).astype(np.float32)
            return result
        

        if key == self.KEYS.TENSOR.PSF_MATRIX:
            c = self.config('psf')
            result = {}

            dataset = sio.loadmat(c['path_file'])
            data = dataset[c['path_dataset']]
            # result = sparse.csc_matrix(data)
            result = sparse.coo_matrix(data)
            print('the psf matrix shape is:', result.shape)
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
                KT.LORS: self.load_local_data(KT.LORS),
                KT.PSF_MATRIX : self.load_local_data(KT.PSF_MATRIX)
            }

            wg = WorkerGraphPSF(mg.tensor(MKT.X), mg.tensor(MKT.BUFFER)[self.task_index], mg.tensor(MKT.SUBSET),
                                inputs=inputs, task_index=self.task_index, name=self.name / 'worker_{}'.format(self.task_index))
            self.subgraphs[KS.WORKER][self.task_index] = wg
            logger.info("Worker graph {} created.".format(self.task_index))
        else:
            logger.info("Skip make worker graph in master process.")
