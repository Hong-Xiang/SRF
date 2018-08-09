from dxl.learn.core import Graph, Tensor, Variable, Constant, NoOp
from srf.model.map_step import MapStep
import tensorflow as tf
from srf.tensor import Image
from dxl.learn.function import ControlDependencies
from srf.preprocess.function.on_tor_lors import Axis as AXIS
import numpy as np

from dxl.core.debug import enter_debug
enter_debug()
from dxl.learn.core import ThisSession



class RingEfficiencyMap(Graph):
    class KEYS(Graph.KEYS):
        class TENSOR(Graph.KEYS.TENSOR):
            X = 'x'
            LORS = 'lors'
            LORS_VALUE = 'lors_value'
            INIT = 'init'
            RESULT = 'result'

        class GRAPH(Graph.KEYS.GRAPH):
            MAP_STEP = 'map_step'

    def __init__(self, info, *, compute_graph, lors, grid, center, size, config=None):

        self.lors = lors
        super().__init__(info,
                         graphs={
                             self.KEYS.GRAPH.MAP_STEP: compute_graph},
                         config={
                             'grid': grid,
                             'center': center,
                             'size': size,
                         })
        # print('center:!!!!!!!!!!PRE:' ,self.config('center'))

    def _load_lors_and_value(self):
        """
        """
        from srf.physics import SplitLorsModel
        if isinstance(self.graphs[self.KEYS.GRAPH.MAP_STEP].graphs['backprojection'].physical_model, SplitLorsModel):
            Axis = ('x', 'y', 'z')
            lors = {i: Constant(self.lors[j].astype(np.float32), 'lors_{}'.format(i))
                    for i, j in zip(Axis, AXIS)}
            lors_value_array = {i: np.ones(
                [self.lors[j].shape[0], 1], dtype=np.float32) for i, j in zip(Axis, AXIS)}
            lors_value = {i: Constant(
                lors_value_array[i], 'lors_value_{}'.format(i)) for i in Axis}
        else:
            print(self.lors[0])
            lors = Constant(self.lors.astype(np.float32), 'lors')
            lors_value =  Constant( np.ones([self.lors.shape[0], 1], dtype=np.float32), 'lors_value' )
        return {'lors': lors, 'lors_value': lors_value}

    def _construct_inputs(self):
        """
        """
        KT = self.KEYS.TENSOR
        with tf.variable_scope('local_inputs'):
            lors_inputs = self._load_lors_and_value()
            self.tensors[KT.X] = Variable(self.info.child_tensor(KT.X),
                                          initializer=np.zeros(self.config('grid'), dtype=np.float32))
            # to_init = [self.tensors[KT.X],
            with ControlDependencies([self.tensors[KT.X].init().data, ]):
                self.tensors[KT.INIT] = NoOp()
        # print('center:!!!!!!!!!!:' ,self.config('center'))
        inputs = {'image': Image(self.tensors[KT.X],
                                 self.config('center'),
                                 self.config('size')),
                  }
        inputs.update(lors_inputs)
        return inputs

    def _construct_x_results(self, inputs):
        KS, KT = self.KEYS.GRAPH, self.KEYS.TENSOR
        self.tensors[KT.RESULT] = self.graphs[KS.MAP_STEP](inputs)

    def kernel(self, inputs=None):
        inputs = self._construct_inputs()
        self._construct_x_results(inputs)

        KS = self.KEYS.GRAPH
        c = self.graphs[KS.MAP_STEP]
        c.make()

    def run(self, sess=None):
        KT = self.KEYS.TENSOR
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        # with tf.Session(config=config) as sess:
        ThisSession.run(self.tensors[KT.INIT])
        # ThisSession.run(self.tensors[KT.MAP_STEP])
        result = ThisSession.run(self.tensors[KT.RESULT])

        # tf.reset_default_graph()
        return result
