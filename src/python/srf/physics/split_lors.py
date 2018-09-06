import tensorflow as tf
from dxl.learn.core import ConfigurableWithName, Tensor
from doufo.tensor import sum_
import os

# load op
TF_ROOT = os.environ.get('TENSORFLOW_ROOT')


class Op:
    op = None

    @classmethod
    def load(cls):
        cls.op = tf.load_op_library(
            TF_ROOT + '/bazel-bin/tensorflow/core/user_ops/tof_tor.so')

    @classmethod
    def get_module(cls):
        if cls.op is None:
            cls.load()
        return cls.op


class SplitLorsModel(ConfigurableWithName):
    """
    This model provides support to the models (typically for tor model) using split lors.
    In these model, lors are split into group and processed respectively.
    """
    class KEYS:
        KERNEL_WIDTH = 'kernel_width'
        TOF_BIN = 'tof_bin'
        TOF_SIGMA2 = 'tof_sigma2'

    def __init__(self, name, *, kernel_width=None, tof_bin=None, tof_sigma2=None, config=None):
        config = self._parse_input_config(config, {
            self.KEYS.KERNEL_WIDTH: kernel_width,
            self.KEYS.TOF_BIN: tof_bin,
            self.KEYS.TOF_SIGMA2: tof_sigma2
        })
        super().__init__(name, config)

    @classmethod
    def _default_config(self):
        return {
            self.KEYS.KERNEL_WIDTH: 1.0,
            self.KEYS.TOF_BIN: 1.0,
            self.KEYS.TOF_SIGMA2: 1.0,
        }
    AXIS = ('x', 'y', 'z')

    @classmethod
    def perm(cls, axis):
        if axis == 'z':
            return [2, 1, 0]
        if axis == 'y':
            return [1, 2, 0]
        if axis == 'x':
            return [0, 2, 1]

    @classmethod
    def perm_back(cls, axis):
        if axis == 'z':
            return [2, 1, 0]
        if axis == 'y':
            return [2, 0, 1]
        if axis == 'x':
            return [0, 2, 1]

    # def rotate_param(self, value, axis):
    #     return [value[p] for p in self.perm(axis)]

    def projection(self, image, lors):
        result = {}
        for a in self.AXIS:
            lors_axis = lors[a].transpose()
            image_axis = image.transpose(self.perm(a))
            result[a] = Tensor(Op.get_module().projection(
                lors=lors_axis.data,
                image=image_axis.data,
                grid=image_axis.grid[::-1],
                center=image_axis.center[::-1],
                size=image_axis.size[::-1],
                kernel_width=self.config(self.KEYS.KERNEL_WIDTH),
                tof_bin=self.config(self.KEYS.TOF_BIN),
                tof_sigma2=self.config(self.KEYS.TOF_SIGMA2)))
        return result

    def check_inputs(self, data, name):
        if not isinstance(data, dict):
            raise TypeError(
                "{} should be dict, got {}.".format(name, data))
        for a in self.AXIS:
            if not a in data:
                raise ValueError("{} missing axis {}.".format(name, a))

    def backprojection(self, lors, image):
        lors_value = lors['lors_value']
        lors = lors['lors']
        result = {}
        for a in self.AXIS:
            lors_axis = lors[a].transpose()
            lors_value_axis = lors_value[a]
            image_axis = image.transpose(self.perm(a))
            backproj = Tensor(Op.get_module().backprojection(
                image=image_axis.data,
                grid=image_axis.grid[::-1],
                center=image_axis.center[::-1],
                size=image_axis.size[::-1],
                lors=lors_axis.data,
                lors_value=lors_value_axis.data,
                kernel_width=self.config(self.KEYS.KERNEL_WIDTH),
                tof_bin=self.config(self.KEYS.TOF_BIN),
                tof_sigma2=self.config(self.KEYS.TOF_SIGMA2)))
            result[a] = backproj.transpose(self.perm_back(a))

        result = sum_(self.name / 'summation')(result)
        return result

    def map_lors(self, lors, image):
        lors_value = lors['lors_value']
        lors = lors['lors']
        # print('lors:', lors)
        # print('lors_value:', lors_value)
        result = {}
        for a in self.AXIS:
            lors_axis = lors[a].transpose()
            lors_value_axis = lors_value[a]
            # print("before trans:",image.size)
            image_axis = image.transpose(self.perm(a))
            # print("before op:",image_axis.size)
            backproj = Tensor(Op.get_module().maplors(
                image=image_axis.data,
                grid=image_axis.grid[::-1],
                center=image_axis.center[::-1],
                size=image_axis.size[::-1],
                lors=lors_axis.data,
                lors_value=lors_value_axis.data,
                kernel_width=self.config(self.KEYS.KERNEL_WIDTH)))
            result[a] = backproj.transpose(self.perm_back(a))
            # print(result[a].shape)
        result = sum_(self.name / 'summation')(result)
        return result
