from dxl.learn import Graph
from dxl.learn.function import dependencies
from dxl.learn.tensor import no_op

from srf.data import Image


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

        class GRAPH(Graph.KEYS.GRAPH):
            RECONSTRUCTION = 'reconstruction'

    def __init__(self, name, work_step, x=None, x_target=None, loader=None, task_index=None):
        super().__init__(name)
        self.tensors[self.KEYS.TENSOR.X] = x
        self.tensors[self.KEYS.TENSOR.TARGET] = x_target
        self.work_step = work_step
        self.config[self.KEYS.CONFIG.TASK_INDEX] = task_index
        self.loader = loader

    def kernel(self, inputs=None):
        inputs = self._construct_inputs()
        result = self._construct_x_result(inputs)
        self._construct_x_update(result)

    @property
    def task_index(self):
        return self.config('task_index')

    def _construct_inputs(self):
        # TODO a better mechnism than using inputs
        local_inputs, local_inputs_init = self.loader.load(self)
        for k, v in local_inputs.items():
            self.tensors[k] = v
        with dependencies(local_inputs_init):
            self.tensors[self.KEYS.TENSOR.INIT] = no_op()
        inputs = {'image': self.tensors[self.KEYS.TENSOR.X],
                  self.KEYS.TENSOR.TARGET: self.tensors[self.KEYS.TENSOR.TARGET]}
        inputs.update(local_inputs)
        return inputs

    def _construct_x_result(self, inputs):
        result = self.tensors[self.KEYS.TENSOR.RESULT] = self.work_step(inputs)
        return result

    def _construct_x_update(self, result):
        """
        update the master x buffer with the x_result of workers.
        """
        KT = self.KEYS.TENSOR
        self.tensors[KT.UPDATE] = self.tensors[KT.TARGET].assign(self.tensors[KT.RESULT].data)


class OSEMWorkerGraph(WorkerGraph):
    class KEYS(WorkerGraph.KEYS):
        class CONFIG(WorkerGraph.KEYS.CONFIG):
            NB_SUBSETS = 'nb_subsets'

        class TENSOR(WorkerGraph.KEYS.TENSOR):
            SUBSET = 'subset'

    def __init__(self, name, work_step, x, x_target, subset, loader, nb_subsets=None):
        super().__init__(name, work_step, x, x_target, loader=loader)
        self.tensors[self.KEYS.TENSOR.SUBSET] = subset
        self.config.update(self.KEYS.CONFIG.NB_SUBSETS, nb_subsets)

    def _construct_inputs(self):
        KC, KT = self.KEYS.CONFIG, self.KEYS.TENSOR
        inputs = super()._construct_inputs()
        for k in self._loader.to_split(self):
            inputs[k] = inputs[k].split_with_index(
                self.config(KC.NB_SUBSETS), self.tensor(KT.SUBSET))
        return inputs
