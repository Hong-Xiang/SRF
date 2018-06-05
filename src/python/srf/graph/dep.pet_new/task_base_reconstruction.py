from dxl.learn.core import make_distribute_session, barrier_single
from dxl.learn.graph import MasterWorkerTaskBase
from dxl.learn.core import Master, Barrier, ThisHost, ThisSession, Tensor, barrier_single
from dxl.learn.core.tensor import NoOp

from dxl.data.io import load_array
from srf.graph.pet_new.master_osem import OsemMasterGraph
from srf.graph.pet_new.worker_osem import OsemWorkerGraph

import json
import numpy as np
import logging
from tqdm import tqdm

logger = logging.getLogger('srf')


class ReconstructionTaskBase(MasterWorkerTaskBase):
    """ A ReconstructionTaskBase is a compute graph template.

    ReconstrutionTaskBase derived class from MasterWorkerTaskBase class in dxl-learn.
    A general reconstruction task need to create master and worker graphs.
    The master graph is usually common, but the worker graph is different, depending on
    the special algorithm and projection models. So a worker_graph_cls is used to specify
    the worker graph class.


    """
    master_graph_cls = OsemMasterGraph
    worker_graph_cls = OsemWorkerGraph

    class KEYS(MasterWorkerTaskBase.KEYS):
        class CONFIG(MasterWorkerTaskBase.KEYS.CONFIG):
            # NB_SUBSETS = 'nb_subsets'
            EFFICIENCY_MAP = 'efficiency_map'

        class TENSOR(MasterWorkerTaskBase.KEYS.TENSOR):
            X = 'x'
            EFFICIENCY_MAP = 'efficiency_map'
            INIT = 'init'
            RECON = 'recon'
            MERGE = 'merge'

        class SUBGRAPH(MasterWorkerTaskBase.KEYS.SUBGRAPH):
            pass

    def __init__(self, task_spec, name=None, graph_info=None, *,
                 job=None, task_index=None, cluster_config=None):

        if name is None:
            name = 'distribute_reconstruction_task'
        super().__init__(job=job, task_index=task_index, cluster_config=cluster_config, name=name,
                         graph_info=graph_info, config=task_spec)

    def load_local_data(self, key) ->np.ndarray:
        """load the local input data.
        called by _make_master_graph and _make_worker_graph.
        """
        KC = self.KEYS.CONFIG
        if key == self.KEYS.TENSOR.X:
            shape = self.config('image')['grid']
            return np.ones(shape, dtype=np.float32)
        if key == self.KEYS.TENSOR.EFFICIENCY_MAP:
            return load_array(self.config(KC.EFFICIENCY_MAP)).astype(np.float32)
        raise KeyError("Unknown key for load_local_data: {}.".format(key))

    def _make_master_graph(self):
        """Make the computation graph runing on the master node.

        A template method that can be rewrote by users.
        Define the task of master node for general PET reconstruction.

        Main functions:
        1. Create a MasterGraph and add it into the subgraph of this task.
        2. Specify a main tensor of this graph (e.g. the result image).
        3. Specify and create the image tensor of a reconstruction task.
        """
        mg = self.master_graph_cls(
            self.load_local_data(self.KEYS.TENSOR.X), name=self.name / 'master')
        self.subgraphs[self.KEYS.SUBGRAPH.MASTER] = mg
        self.tensors[self.KEYS.TENSOR.MAIN] = mg.tensor(self.KEYS.TENSOR.X)
        self.tensors[self.KEYS.TENSOR.X] = mg.tensor(self.KEYS.TENSOR.X)
        logger.info('Master graph created')

    def _make_worker_graphs(self):
        """Make the computation graph running on the worker node.

        A template method that can be rewrote by users.
        Define the task of master node for general PET reconstruction.

        Main functions:
        1. Create computation graphs of workers and add them into the 
           subgraph list of the reconstruction task.
        2. Specify the key tensors for workers.
        """
        KS = self.KEYS.SUBGRAPH
        if not ThisHost.is_master():
            self.subgraphs[KS.WORKER] = [
                None for i in range(self.nb_workers)]
            mg = self.subgraph(KS.MASTER)
            MKT = mg.KEYS.TENSOR
            inputs = {
                self.KEYS.TENSOR.EFFICIENCY_MAP: self.load_local_data(
                    self.KEYS.TENSOR.EFFICIENCY_MAP)
            }
            wg = self.worker_graph_cls(mg.tensor(MKT.X), mg.tensor(MKT.BUFFER)[self.task_index], mg.tensor(MKT.SUBSET),
                                       inputs=inputs, task_index=self.task_index, name=self.name / 'worker_{}'.format(self.task_index))
            self.subgraphs[KS.WORKER][self.task_index] = wg
            logger.info("Worker graph {} created.".format(self.task_index))
        else:
            logger.info("Skip make worker graph in master process.")

    def _make_init_barrier(self):
        """Create the initial barrier 
        This barrier ensure the initialization of master and worker has been finished 
        between the reconstruction.
        """
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
        """Create the reconstruction barrier
        Make sure the recosntruction of workers has been completed.
        """
        # mg = self.subgraph(self.KEYS.SUBGRAPH.MASTER)
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
        """create the image summation  barrier.
        Make sure the image has been merged and updated before next iteration.
        """
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
        """ Rewrite template method to create specific barriers.
        """
        self._make_init_barrier()
        self._make_recon_barrier()
        self._make_merge_barrier()

    def _run_with_info(self, key):
        # logger.info('Start {}...'.format(key))
        ThisSession.run(self.tensor(key))
        # logger.info('{} Complete.'.format(key))

    def run_task(self):
        KT = self.KEYS.TENSOR
        # KC = self.KEYS.CONFIG
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
                # self._print_x()
                self._save_result('{}_{}_{}.npy'.format(image_name, i, j))

    @MasterWorkerTaskBase.master_only
    def _save_result(self, path):
        x = ThisSession.run(self.tensor(self.KEYS.TENSOR.X))
        np.save(path, x)

    def _print_x(self):
        x = ThisSession.run(self.tensor(self.KEYS.TENSOR.X))
        print('X value:', x, sep='\n')
