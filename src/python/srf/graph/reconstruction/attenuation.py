import numpy as np
import tensorflow as tf
from dxl.learn import Graph
from dxl.learn.session import ThisSession
from dxl.learn.tensor import const, variable_from_tensor, initializer

from srf.data import Image


class Attenuation(Graph):
    class KEYS(Graph.KEYS):
        class TENSOR(Graph.KEYS.TENSOR):
            X = 'x'
            LORS = 'lors'
            LORS_VALUE = 'lors_value'
            INIT = 'init'
            RESULT = 'result'

        class GRAPH(Graph.KEYS.GRAPH):
            ATTEN_STEP = 'attenuation'

    def __init__(self, name, projection, projection_data, image, center, size):
        super().__init__(name),
        self.image = image
        self.config.update('center', center)
        self.config.update('size', size)
        self.projection_data = projection_data
        self.projection = projection

    def _construct_inputs(self):
        x = self.tensors[self.KEYS.TENSOR.X] = const[tf](self.image, name='const')
        return self.projection_data, Image(x, self.config['center'], self.config['size'])

    def _construct_x_results(self, projection_data, image):
        self.tensors[self.KEYS.TENSOR.RESULT] = self.projection(image,projection_data)

    def kernel(self):
        self._construct_x_results(*self._construct_inputs())

    def run(self, session=None):
        if session is None:
            session = ThisSession
        return session.run(self.tensors[self.KEYS.TENSOR.RESULT].values)
