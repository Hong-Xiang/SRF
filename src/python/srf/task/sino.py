import json, h5py, time, logging
import numpy as np
from tqdm import tqdm
from scipy import sparse

from dxl.learn.core import Barrier, make_distribute_session
from dxl.learn.core import Master, Barrier, ThisHost, ThisSession, Tensor

from ..graph.master_sino import MasterGraph
from ..graph.worker_sino import WorkerGraphSINO
from ..services.utils import print_tensor, debug_tensor
from ..preprocess import preprocess_sino 

from ..app.sinoapp import logger

from .sinodata import ImageInfo, ReconInfo, InputInfo, SinoInfo, MatrixInfo

from .srftask import SRFTask


# sample_reconstruction_config = {
#     'grid': [150, 150, 150],
#     'center': [0., 0., 0.],
#     'size': [150., 150., 150.],
#     'map_file': './debug/map.npy',
#     'sino_files': './debug/sino.npz',      #.npy是二维矩阵数组？？？
#     'system_matrix':'./debug/matrix.npy'
# }



class SinoTask(SRFTask):
    class KEYS(SRFTask.KEYS):
        class STEPS(SRFTask.KEYS.STEPS):
            INIT = 'init_step'
            RECON = 'recon_step'
            MERGE = 'merge_step'

    def __init__(self, job, task_index, task_configs, distribute_configs):
        super().__init__(job, task_index, task_configs, distribute_configs)
        # self.steps = {}

    def parse_task(self):
        super().parse_task()
        ti = self.task_info
        ii = ti['Image_info']
        self.image_info = ImageInfo(ii['grid'],
                                    ii['center'],
                                    ii['size'],
                                    ii['filename'],
                                    ii['map_filename'])
        # self.kernel_width = ti['kernel_width']

        ri = ti['Recon_info']
        self.Reconinfo = ReconInfo(ri['nb_iterations'],
                                  ri['nb_subsets'],
                                  ri['save_interval'])

        Ii = ti['Input_info']
        self.Inputinfo = InputInfo(Ii['Input_file'],
                                   Ii['system_matrix'])

    def pre_works(self):
        nb_workers = self.nb_workers()
        nb_subsets = self.Reconinfo.nb_subsets
        filedir = self.work_directory + self.Inputinfo.Input_file
        #print('filedir:', filedir)
        Input = np.load(filedir)
        sino = preprocess_sino.preprocess_sino(Input)
        #print(sino[443877])

        filedir2 = self.work_directory + self.Inputinfo.sm
        matrix = np.load(filedir2)

        # compute the lors shape and step of a subset in OSEM.
        #sino_files =  self.work_directory +self.Inputinfo.Input_file 
        sino_steps = sino.shape[0]//(nb_workers*nb_subsets)
        sino_shapes = [sino_steps, 1]
        self.sino_info = SinoInfo(
            filedir,
            sino_shapes,
            sino_steps,
            None
        )
        matrix_steps = matrix.shape[0]//(nb_subsets*nb_workers)
        matrix_shapes = [matrix_steps, matrix.shape[1]]
        self.matrix_info = MatrixInfo(
            filedir2,
            matrix_shapes,
            matrix_steps
        )

        image_grid = self.image_info.grid
        new_grid = [image_grid[0]*image_grid[1]*image_grid[2],1]
        self.image_info = ImageInfo(
            new_grid,
            self.image_info.center,
            self.image_info.size,
            self.image_info.name,
            self.image_info.map_file
        )



    def create_master_graph(self):
        '''
        attention:nb_workers here depends on size of sinogram
        '''
        x = np.ones(self.image_info.grid, dtype=np.float32)
        mg = MasterGraph(x, self.nb_workers(), self.ginfo_master())
        self.add_master_graph(mg)
        logger.info("Global graph created.")
        return mg

    def create_worker_graphs(self):
        for i in range(self.nb_workers()):
            logger.info("Creating local graph for worker {}...".format(i))
            self.add_worker_graph(
                WorkerGraphSINO(
                    self.master_graph,
                    self.image_info,
                    self.sino_info,
                    self.matrix_info,
                    i,
                    self.ginfo_worker(i),
                ))
        logger.info("All local graph created.")
        return self.worker_graphs

    def bind_local_data(self):
        """
        bind the static effmap data
        """
        map_file = self.work_directory + self.image_info.map_file
        #matrix_file = self.work_directory + self.Inputinfo.sm
        task_index = ThisHost.host().task_index
        if ThisHost.is_master():
            logger.info("On Master node, skip bind local data.")
            return
        else:
            logger.info(
                "On Worker node, local data for worker {}.".format(task_index))
            emap = self.load_local_effmap(map_file)
            #print(emap.shape)
            #matrix = self.load_local_matrix(matrix_file)
            self.worker_graphs[task_index].init_efficiency_map(emap)
        self.bind_local_sino()
        self.bind_local_matrix()      
    
    
    def bind_local_sino(self, task_index=None):
        if task_index is None:
            task_index = ThisHost.host().task_index
        if ThisHost.is_master():
            logger.info("On Master node, skip bind local sino.")
            return
        else:
            logger.info(
                "On Worker node, local data for worker {}.".format(task_index))
            worker_sinos = self.load_local_sino(task_index)
            worker_sino = preprocess_sino.preprocess_sino(worker_sinos)
            #worker_sino = sparse.csr_matrix(worker_sino)
            #self.worker_graphs[task_index].assign_efficiency_map(emap)
            self.worker_graphs[task_index].assign_sinos(worker_sino,
                                                       self.Reconinfo.nb_subsets)
    

    def bind_local_matrix(self,task_index=None):
        if task_index is None:
            task_index = ThisHost.host().task_index
        if ThisHost.is_master():
            logger.info("On Master node, skip bind local matrix.")
            return
        else:
            logger.info("On Worker node, local data for worker {}.".format(task_index))
            worker_matrix = self.load_local_matrix(task_index)
            #worker_matrix = sparse.coo_matrix((worker_matrix[2,:],(worker_matrix[0,:],worker_matrix[1,:])),shape=(640000,40*40*3))
            #worker_matrix = sparse.coo_matrix(worker_matrix)
            self.worker_graphs[task_index].assign_matrixs(worker_matrix,self.Reconinfo.nb_subsets)


    def make_steps(self):
        KS = self.KEYS.STEPS
        self._make_init_step(KS.INIT)
        self._make_recon_step(KS.RECON)
        self._make_merge_step(KS.MERGE)
        # self.steps = {
        #     KS.INIT: init_step,
        #     KS.RECON: recon_step,
        #     KS.MERGE: merge_step,
        # }
           
    def _make_init_step(self, name='init'):
        init_barrier = Barrier(name, self.hosts, [self.master_host],
                                [[g.tensor(g.KEYS.TENSOR.INIT)]
                                for g in self.worker_graphs])
        master_op = init_barrier.barrier(self.master_host)
        worker_ops = [init_barrier.barrier(h) for h in self.hosts]
        self.add_step(name, master_op, worker_ops)        ###What is the function, why all three steps have such function
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

    def _assign_sinos(self, subset_index):
        if ThisHost.is_master():
            pass
        else:
            task_index = ThisHost.host().task
            this_worker = self.worker_graphs[task_index]
            ThisSession.run(this_worker.tensor(
                this_worker.KEYS.TENSOR.ASSIGN_SINOS)[subset_index].data)

    def _assign_matrixs(self, subset_index):
        if ThisHost.is_master():
            pass
        else:
            task_index = ThisHost.host().task
            this_worker = self.worker_graphs[task_index]
            ThisSession.run(this_worker.tensor(
                this_worker.KEYS.TENSOR.ASSIGN_MATRIXS)[subset_index].data)


    def run(self):
        KS = self.KEYS.STEPS
        self.run_step_of_this_host(KS.INIT)
        logger.info('STEP: {} done.'.format(KS.INIT))
        nb_iterations = self.Reconinfo.nb_iterations
        nb_subsets = self.Reconinfo.nb_subsets
        image_name = self.image_info.name
        for i in tqdm(range(nb_iterations), ascii=True):
            for j in tqdm(range(nb_subsets), ascii=True):
                print("start assign sinos !!!!!!!")
                self._assign_sinos(j)
                self._assign_matrixs(j)
                logger.info('STEP: {} {} done.'.format('assign', j))

                        
                self.run_step_of_this_host(KS.RECON)
                logger.info('STEP: {} done.'.format(KS.RECON))

                self.run_step_of_this_host(KS.MERGE)
                logger.info('STEP: {} done.'.format(KS.MERGE))

                self.run_and_print_if_not_master(
                    self.worker_graphs[ThisHost.host().task].tensor(self.worker_graphs[0].KEYS.TENSOR.RESULT)
                )
                self.run_and_print_if_is_master(
                    self.master_graph.tensor('x')
                )
                
                self.run_and_save_if_is_master(
                    self.master_graph.tensor('x'),
                    image_name+'_{}_{}.npy'.format(i, j))
        logger.info('Recon {} steps done.'.format(nb_iterations,nb_subsets))
        #time.sleep(5)
    


    def run_and_save_if_is_master(self, x, path):
        if ThisHost.is_master():
            if isinstance(x, Tensor):
                x = x.data
            result = ThisSession.run(x)
            np.save(path, result)

    def run_and_print_if_not_master(self, x):
        if not ThisHost.is_master():
            if isinstance(x, Tensor):
                x = x.data
            result = ThisSession.run(x)
            #print(result)
    def run_and_print_if_is_master(self, x):
        if ThisHost.is_master():
            if isinstance(x, Tensor):
                x = x.data
            result = ThisSession.run(x)
            print(result)

    def load_data(self, file_name, range = None):
        if file_name.endswith('.npy'):
            data = np.load(file_name)
        elif file_name.endswith('.h5'):
            with h5py.File(file_name, 'r') as fin:
                data = np.array(fin['data'])
        return data

    # Load datas
    def load_local_sino(self, task_index: int):

        #sino = {}
        NS = self.Reconinfo.nb_subsets
        SI = self.sino_info
        worker_step =  SI.sino_steps() * NS
        print(worker_step)
        # print("!!!!!!!!!!!!!", task_index)
        sino_ranges = [task_index * worker_step,
                           (task_index+1) * worker_step] 
        
        msg = "Loading sinos from file: {}"
        logger.info(msg.format(SI.sino_file()))
        sino = self.load_data(SI.sino_file(),sino_ranges)
        logger.info('Loading local data done.')
        return sino

    def load_local_effmap(self, map_file):
        logger.info("Loading efficiency map from file: {}...".format(map_file))
        emap = self.load_data(map_file)
        return emap

    def load_local_matrix(self,task_index:int):
        NS = self.Reconinfo.nb_subsets
        MI = self.matrix_info
        worker_step = MI.matrix_steps() * NS
        matrix_ranges = [task_index * worker_step,(task_index+1)*worker_step]
        logger.info("Loading system matrix from file: {}...".format(MI.matrix_file()))
        mat = self.load_data(MI.matrix_file())
        return mat

