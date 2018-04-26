# from .utils import constant_tensor
from dxl.learn.core import Master, Graph, Tensor, tf_tensor, variable
import tensorflow as tf


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
            ASSIGN_SINOS = 'ASSIGN_SINOS'
            ASSIGN_MATRIXS = 'ASSIGN_MATRIXS'

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

    def _construct_inputs(self):
        KT = self.KEYS.TENSOR
        self.tensors[KT.EFFICIENCY_MAP] = variable(
            self.graph_info.update(name='effmap_{}'.format(self.task_index)),
            None,
            self.tensor(self.KEYS.TENSOR.X).shape,
            tf.float32)


        SI = self.sino_info
        self.tensors[KT.SINOS] = variable(
            self.graph_info.update(name='sino_{}'.format(self.task_index)),
            None,
            SI.sino_shape(),
            tf.float32)
            
        MI = self.matrix_info
        self.tensors[KT.MATRIX] = variable(
            self.graph_info.update(name='matrix_{}'.format(self.task_index)),
            None,
            MI.matrix_shape(),
            tf.float32)

        self.tensors[KT.INIT] = Tensor(
            tf.no_op(), None, self.graph_info.update(name='init_no_op'))


    def assign_sinos(self,worker_sinos,nb_subsets):
        assign_sinos = []
        SI = self.sino_info
        for i in range(nb_subsets):
            assign_current_subset = self.tensor(self.KEYS.TENSOR.SINOS).assign(
                worker_sinos[i*SI.sino_steps() :(i+1)*SI.sino_steps()])
            with tf.control_dependencies([assign_current_subset.data]):
                init = tf.no_op()
            assign_sinos.append(Tensor(
                init, None, self.graph_info.update(name='assign_sino_subset_{}'.format(i))))
        self.tensors[self.KEYS.TENSOR.ASSIGN_SINOS] = assign_sinos

    def init_efficiency_map(self, efficiency_map):
        map_assign = self.tensor(
            self.KEYS.TENSOR.EFFICIENCY_MAP).assign(efficiency_map)
        with tf.control_dependencies([map_assign.data]):
            init = tf.no_op()
        init = Tensor(init, None, self.graph_info.update(name='init'))
        self.tensors[self.KEYS.TENSOR.INIT] = init

    def assign_matrixs(self,worker_matrixs,nb_subsets):
        assign_matrixs = []
        MI = self.matrix_info
        for i in range(nb_subsets):
            assign_current_subset = self.tensor(self.KEYS.TENSOR.MATRIX).assign(
                worker_matrixs[i*MI.matrix_steps():(i+1)*MI.matrix_steps()])
            with tf.control_dependencies([assign_current_subset.data]):
                init = tf.no_op()
            assign_matrixs.append(Tensor(
                init, None, self.graph_info.update(name='assign_matrix_subset_{}'.format(i))))
        self.tensors[self.KEYS.TENSOR.ASSIGN_MATRIXS] = assign_matrixs

    def _construct_x_result(self):
        self._construct_inputs()
        KT = self.KEYS.TENSOR
        from ..model.sinogram import ReconStep
        x_res = ReconStep(
            'recon_step_{}'.format(self.task_index),
            self.tensor(KT.X, is_required=True),
            self.tensor(KT.EFFICIENCY_MAP, is_required=True),
            self.tensor(KT.SINOS, is_required=True),
            self.tensor(KT.MATRIX, is_required=True),
            self.graph_info.update(name=None))()
        self.tensors[KT.RESULT] = x_res
