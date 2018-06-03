from dxl.learn.core import Graph, Tensor, Variable, Constant, NoOp
from dxl.learn.distribute import DistributeGraphInfo, Host, Master, JOB_NAME
import numpy as np
import tensorflow as tf


class WorkerGraph(Graph):
    class KEYS(Graph.KEYS):
        class CONFIG(Graph.KEYS.CONFIG):
            TASK_INDEX = 'task_index'

        class TENSOR(Graph.KEYS.TENSOR):
            X = 'x'
            TARGET = 'target'
            RESULT = 'result'
            INIT = 'init'
            UPDATE = 'update'

        class SUBGRAPH(Graph.KEYS.SUBGRAPH):
            RECONSTRUCTION = 'reconstruction'

    def __init__(self, info,
                 x: Tensor,
                 x_target: Variable,
                 *,
                 local_inputs_loader,
                 tensors=None,
                 subgraphs=None,
                 config=None):
        config = self._parse_input_config(config, {
            self.KEYS.CONFIG.NB_SUBSETS: nb_subsets,
        })
        self._local_inputs_loader = local_inputs_loader
        super().__init__(info, tensors={
            self.KEYS.TENSOR.X: x,
            self.KEYS.TENSOR.TARGET: x_target,
        }, config=config)

    def kernel(self):
        inputs = self._construct_inputs()
        result = self._construct_x_result(inputs)
        self._construct_x_update(result)

    @property
    def task_index(self):
        return self.config('task_index')

    def _construct_inputs(self):
        KT = self.KEYS.TENSOR
        with tf.variable_scope('local_inputs'):
            local_inputs, local_inputs_init = self._local_inputs_loader.load()
            with tf.control_dependencies([t.data for t in local_inputs_init]):
                self.tensors[self.KEYS.TENSOR.INIT] = NoOp()
        inputs = {KT.X: self.tensor(KT.X), KT.TARGET: self.tensor(KT.TARGET)}
        inputs.update(local_inputs)
        return inputs

    def _construct_x_result(self, inputs):
        KS, KT = self.KEYS.SUBGRAPH, self.KEYS.TENSOR
        reconstruction = self.subgraph(KS.RECONSTRUCTION, SubgraphBuilder.get(
            self.info.name / KS.RECONSTRUCTION)(inputs))
        result = reconstruction()
        self.tensors[KT.RESULT] = result / self.tensor(KT.EFFICIENCY_MAP)

    def _construct_x_update(self):
        """
        update the master x buffer with the x_result of workers.
        """
        KT = self.KEYS.TENSOR
        self.tensors[KT.UPDATE] = self.tensor(
            KT.TARGET).assign(self.tensor(KT.RESULT))


class WorkerGraphToR(WorkerGraph):
    class KEYS(WorkerGraph.KEYS):
        class TENSOR(WorkerGraph.KEYS.TENSOR):
            LORS = 'lors'
            ASSIGN_LORS = 'ASSIGN_LORS'

        class CONFIG(WorkerGraph.KEYS.CONFIG):
            KERNEL_WIDTH = 'kernel_width'
            IMAGE_INFO = 'image'
            TOF_BIN = 'tof_bin'
            TOF_SIGMA2 = 'tof_sigma2'
            TOF_RES = 'tof_res'
            LORS_INFO = 'lors_info'
            TOF = 'tof'

        class SUBGRAPH(WorkerGraph.KEYS.SUBGRAPH):
            RECON_STEP = 'recon_step'

    AXIS = ('x', 'y', 'z')
    REQURED_INPUTS = ()

    def __init__(self,
                 x, x_target, subset, inputs, task_index,
                 nb_subsets=1,
                 kernel_width=None,
                 image_info=None,
                 tof_bin=None,
                 tof_sigma2=None,
                 lors_info=None,
                 graph_info=None,
                 name=None):
        super().__init__(x, x_target, subset, inputs, task_index=task_index, name=name,
                         config={
                             self.KEYS.CONFIG.KERNEL_WIDTH: kernel_width,
                             self.KEYS.CONFIG.IMAGE_INFO: image_info,
                             self.KEYS.CONFIG.TOF_BIN: tof_bin,
                             self.KEYS.CONFIG.TOF_SIGMA2: tof_sigma2,
                             self.KEYS.CONFIG.LORS_INFO: lors_info,
                         }, graph_info=graph_info)

    def _construct_inputs(self, inputs):
        KT = self.KEYS.TENSOR
        effmap_name = KT.EFFICIENCY_MAP
        super()._construct_inputs({effmap_name: inputs[effmap_name]})
        self.tensors[KT.LORS] = {a: self.process_lors(
            inputs[KT.LORS][a], a) for a in self.AXIS}
        self.tensors[KT.INIT] = NoOp()

    def process_lors(self, lor: np.ndarray, axis):
        KT = self.KEYS.TENSOR
        KC = self.KEYS.CONFIG
        step = lor.shape[0] // self.config(KC.NB_SUBSETS)
        # step = lor.shape[0]
        columns = lor.shape[1]
        lor = Constant(lor, None, self.info.child(KT.LORS))
        if self.config(KC.NB_SUBSETS) == 1:
            return lor
        lor = tf.slice(lor.data,
                       [self.tensor(KT.SUBSET).data * step, 0],
                       [step, columns])
        return Tensor(lor, None, self.info.child('{}_{}'.format(KT.LORS, axis)))

    # def assign_lors(self, worker_lors, nb_subsets):
    #     assign_lors = []
    #     LI = self.lors_info
    #     for i in range(nb_subsets):
    #         assign_current_subset = [self.tensor(self.KEYS.TENSOR.LORS)[a].assign(
    #             worker_lors[a][i * LI.lors_steps(a):(i + 1) * LI.lors_steps(a)]) for a in worker_lors]
    #         with tf.control_dependencies([a.data for a in assign_current_subset]):
    #             init = tf.no_op()
    #         assign_lors.append(Tensor(
    #             init, None, self.graph_info.update(name='assign_subset_{}'.format(i))))
    #     self.tensors[self.KEYS.TENSOR.ASSIGN_LORS] = assign_lors

    def _construct_x_result(self):
        KT = self.KEYS.TENSOR
        # self.tensors[KT.RESULT] = self.tensor(KT.X)
        # return
        KC = self.KEYS.CONFIG
        from ...model.tor_step import TorStep
        print(self.config('tof')[KC.TOF_SIGMA2])
        self.subgraphs[self.KEYS.SUBGRAPH.RECON_STEP] = TorStep(
            self.name / 'recon_step_{}'.format(self.task_index),
            self.tensor(KT.X, is_required=True),
            self.tensor(KT.EFFICIENCY_MAP, is_required=True),
            self.config(KC.IMAGE_INFO)['grid'],
            self.config(KC.IMAGE_INFO)['center'],
            self.config(KC.IMAGE_INFO)['size'],
            self.config('tor')[KC.KERNEL_WIDTH],
            self.config('tof')[KC.TOF_BIN],
            self.config('tof')[KC.TOF_SIGMA2],
            self.tensor(KT.LORS)['x'],
            self.tensor(KT.LORS)['y'],
            self.tensor(KT.LORS)['z'],
            self.info.update(name=None))
        x_res = self.subgraph(self.KEYS.SUBGRAPH.RECON_STEP)()
        self.tensors[KT.RESULT] = x_res
