import tensorflow as tf
from dxl.learn.core import ConfigurableWithName, Tensor
import os
from srf.tensor import Image
# load op
TF_ROOT = os.environ.get('TENSORFLOW_ROOT')
# print(TF_ROOT)


class Op:
    # _loaded = None
    op = None

    # @property
    # def backprojection_gpu(self):
    #     if self._loaded is None:
    #         self._loaded = tf.load_op_library(
    #             TF_ROOT + '/bazel-bin/tensorflow/core/user_ops/tor.so')
    #     return self._loaded

    @classmethod
    def load(cls):
        cls.op = tf.load_op_library(
            TF_ROOT + '/bazel-bin/tensorflow/core/user_ops/tor.so')

    @classmethod
    def get_module(cls):
        if cls.op is None:
            cls.load()
        return cls.op

# op = Op()


class ToRMapModel(ConfigurableWithName):
    class KEYS:
        KERNEL_WIDTH = 'kernel_width'

    def __init__(self, name, *, kernel_width=None, config=None):
        config = self._parse_input_config(config, {
            self.KEYS.KERNEL_WIDTH: kernel_width
        })
        super().__init__(name, config)

    @classmethod
    def _default_config(self):
        return {
            self.KEYS.KERNEL_WIDTH: 1.0
        }
    AXIS = ('x', 'y', 'z')

    def perm(self, axis):
        if axis == 'z':
            return [2, 1, 0]
        if axis == 'y':
            return [1, 2, 0]
        if axis == 'x':
            return [0, 2, 1]

    def perm_back(self, axis):
        if axis == 'z':
            return [2, 1, 0]
        if axis == 'y':
            return [2, 0, 1]
        if axis == 'x':
            return [0, 2, 1]

    def rotate_param(self, value, axis):
        return [value[p] for p in self.perm(axis)]

    def check_inputs(self, data, name):
        if not isinstance(data, dict):
            raise TypeError(
                "{} should be dict, got {}.".format(name, data))
        for a in self.AXIS:
            if not a in data:
                raise ValueError("{} missing axis {}.".format(name, a))

    def backprojection(self, lors, image: Image):
        lors_value = lors['lors_value']
        lors = lors['lors']
        lors = lors.transpose()
        result = Tensor(Op.get_module().backprojection_gpu(
            image=image.data,
            grid=image.grid[::-1],
            center=image.center[::-1],
            size=image.size[::-1],
            lors=lors.data,
            lors_value=lors_value.data,
            kernel_width=self.config(self.KEYS.KERNEL_WIDTH)))
        return result
