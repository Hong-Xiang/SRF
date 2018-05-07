# from .utils import constant_tensor
from dxl.learn.core import Master, Graph, Tensor, tf_tensor, variable, tensor
import tensorflow as tf
from dxl.learn.core import ThisHost
from ..app.sinoapp import logger
from ..preprocess import preprocess_sino 
import numpy as np
import scipy.io as sio
from scipy import sparse

class WorkerGraphBase(Graph):
    class KEYS(Graph.KEYS):
        class TENSOR(Graph.KEYS.TENSOR):
            X = 'x'
            UPDATE = 'update'
            RESULT = 'result'

    def __init__(self, global_graph, task_index=None, graph_info=None,
                 name=None):
        self.global_graph = global_graph
        if task_index is None:
            task_index = self.global_graph.host.task_index
        self.task_index = task_index
        if name is None:
            name = 'worker_graph_{}'.format(self.task_index)

        super().__init__(name, graph_info=graph_info)
        self._construct_x()
        self._construct_x_result()
        self._construct_x_update()

    def _construct_x(self):
        x_global = self.global_graph.tensor(self.global_graph.KEYS.TENSOR.X)
        self.tensors[self.KEYS.TENSOR.X] = x_global

    def _construct_x_result(self):
        self.tensors[self.KEYS.TENSOR.RESULT] = self.tensor(self.KEYS.TENSOR.X)

    def _construct_x_update(self):
        """
        update the master x buffer with the x_result of workers.
        """
        x_buffers = self.global_graph.tensor(
            self.global_graph.KEYS.TENSOR.BUFFER)
        x_buffer = x_buffers[self.task_index]
        x_u = x_buffer.assign(self.tensor(self.KEYS.TENSOR.RESULT))
        self.tensors[self.KEYS.TENSOR.UPDATE] = x_u


class WorkerGraphSINO(WorkerGraphBase):
    class KEYS(WorkerGraphBase.KEYS):
        class TENSOR(WorkerGraphBase.KEYS.TENSOR):
            EFFICIENCY_MAP = 'efficiency_map'
            SINOS = 'sino'
            INIT = 'init'
            MATRIX = 'matrix'
            # ASSIGN_SINOS = 'ASSIGN_SINOS'
            # ASSIGN_MATRIXS = 'ASSIGN_MATRIXS'

    def __init__(self,
                 master_graph,
                 image_info,
                 sino_info,
                 matrix_info,
                 task_index,
                 graph_info=None,
                 name=None):
        self.image_info = image_info
        self.sino_info = sino_info
        self.matrix_info = matrix_info
        super().__init__(master_graph, task_index, graph_info, name=name)

    
    def _load_data(self):
        KT = self.KEYS.TENSOR        
        self.init_efficiency_map(self.load_local_effmap(self.image_info.map_file))
        self.init_sino(self.load_local_sino(self.task_index))
        worker_matrix = sparse.coo_matrix(self.load_local_matrix(self.task_index))
        self.init_matrix(worker_matrix)


    def _construct_inputs(self):
        KT = self.KEYS.TENSOR        
        if not ThisHost.is_master() and ThisHost.host().task_index == self.task_index:
            self._load_data()
        else:
            self.tensors[KT.EFFICIENCY_MAP] = None
            self.tensors[KT.SINOS] = None
            self.tensors[KT.MATRIX] = None

            
        
    #     KT = self.KEYS.TENSOR
        # self.tensors[KT.EFFICIENCY_MAP] = variable(
        #     self.graph_info.update(name='effmap_{}'.format(self.task_index)),
        #     None,
        #     self.tensor(self.KEYS.TENSOR.X).shape,
        #     tf.float32)


    #     SI = self.sino_info
    #     self.tensors[KT.SINOS] = variable(
    #         self.graph_info.update(name='sino_{}'.format(self.task_index)),
    #         None,
    #         SI.sino_shape(),
    #         tf.float32)
            
    #     #MI = self.matrix_info
    #     # self.tensors[KT.MATRIX] = variable(
    #     #     self.graph_info.update(name='matrix_{}'.format(self.task_index)),
    #     #     None,
    #     #     MI.matrix_shape(),
    #     #     tf.float32)
    #     #tensors[KT.MATRIX] = tensor.SparseMatrix(None,MI,self.graph_info.update(name='matrix_{}'.format(self.task_index)))

        self.tensors[KT.INIT] = Tensor(
            tf.no_op(), None, self.graph_info.update(name='init_no_op'))


    def init_sino(self, worker_sino):
        sino = tensor.TensorNumpyNDArray(worker_sino.astype(np.float32),graph_info=self.graph_info.update(name='sino_{}'.format(self.task_index)))
        self.tensors[self.KEYS.TENSOR.SINOS] = sino


    def init_efficiency_map(self, efficiency_map):
        eff_map = tensor.TensorNumpyNDArray(efficiency_map.astype(np.float32),graph_info=self.graph_info.update(name='matrix_{}'.format(self.task_index)))
        self.tensors[self.KEYS.TENSOR.EFFICIENCY_MAP] = eff_map

    def init_matrix(self, worker_matrix):
        matrix = tensor.SparseMatrix(worker_matrix,graph_info=self.graph_info.update(name='matrix_{}'.format(self.task_index))) 
        self.tensors[self.KEYS.TENSOR.MATRIX] = matrix
        

    def load_data(self, file_name, range = None):
        if file_name.endswith('.npy'):
            data = np.load(file_name)
        elif file_name.endswith('.h5'):
            with h5py.File(file_name, 'r') as fin:
                data = np.array(fin['data'])
        elif file_name.endswith('.mat'):
            dataset = sio.loadmat(file_name)
            data = dataset['matrix']
        return data

    # Load datas
    def load_local_sino(self, task_index: int):
        #sino = {}
        #NS = self.Reconinfo.nb_subsets
        SI = self.sino_info
        worker_step =  SI.sino_steps()
        sino_ranges = np.zeros(SI.sino_shape()[0],dtype=np.int32)
        for i in range (SI.sino_shape()[0]):
            sino_ranges[i] = i
 

        msg = "Loading sinos from file: {}"
        logger.info(msg.format(SI.sino_file()))
        sino = self.load_data(SI.sino_file())
        logger.info('Loading local data done.')
        sino = preprocess_sino.preprocess_sino(sino)
        sino_index = sino[sino_ranges*worker_step+task_index]
        return sino_index

    def load_local_effmap(self, map_file):
        logger.info("Loading efficiency map from file: {}...".format(map_file))
        emap = self.load_data(map_file)
        return emap

    def load_local_matrix(self,task_index:int):
        #NS = self.Reconinfo.nb_subsets
        MI = self.matrix_info
        SI = self.sino_info
        worker_step =  SI.sino_steps()
        sino_ranges = np.zeros(SI.sino_shape()[0],dtype=np.int32)
        for i in range (SI.sino_shape()[0]):
            sino_ranges[i] = i
        logger.info("Loading system matrix from file: {}...".format(MI.matrix_file()))
        mat = self.load_data(MI.matrix_file())
        mat_index = mat[sino_ranges*worker_step+task_index]
        mat_index = mat_index.astype(np.float32)
        return mat_index

    def _construct_x_result(self):
        self._construct_inputs()
        KT = self.KEYS.TENSOR
        from ..model.sinogram import ReconStep
        if not ThisHost.is_master() and ThisHost.host().task_index == self.task_index:
            x_res = ReconStep(
                'recon_step_{}'.format(self.task_index),
                self.tensor(KT.X, is_required=True),
                self.tensor(KT.EFFICIENCY_MAP, is_required=True),
                self.tensor(KT.SINOS, is_required=True),
                self.tensor(KT.MATRIX, is_required=True),
                self.graph_info.update(name=None))()
            self.tensors[KT.RESULT] = x_res
        else:
            self.tensors[KT.RESULT] = self.tensor(KT.X)
