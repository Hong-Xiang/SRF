
import tensorflow as tf
import os
from dxl.learn.core import ConfigurableWithName, Tensor
from srf.tensor import Image
# load op
TF_ROOT = os.environ.get('TENSORFLOW_ROOT')


op_dir = '/bazel-bin/tensorflow/core/user_ops/'
op_list = {
    'siddon': 'siddon.so',
    'tor': 'tof_tor.so'
}


class Op:
    op = None

    @classmethod
    def load(cls):
        cls.op = tf.load_op_library(
            TF_ROOT + op_dir + 'tof_tor.so')

    @classmethod
    def get_module(cls):
        if cls.op is None:
            cls.load()
        return cls.op


class CompleteLorsModel(ConfigurableWithName):
    """
    This model provides support to the models (typically for siddon model)
    using complete lors.These model processes the lors dataset without splitting.
    """

    class KEYS:
        TOF_BIN = 'tof_bin'
        TOF_SIGMA2 = 'tof_sigma2'

    def __init__(self, name, *, tof_bin=None, tof_sigma2=None, config=None):
        config = self._parse_input_config(config, {
            self.KEYS.TOF_BIN: tof_bin,
            self.KEYS.TOF_SIGMA2: tof_sigma2
        })
        super().__init__(name, config)

    @classmethod
    def _default_config(self):
        return {
            self.KEYS.TOF_BIN: 1.0,
            self.KEYS.TOF_SIGMA2: 1.0,
        }

    def rotate_param(self, value, axis):
        return [value[p] for p in self.perm(axis)]

    def projection(self, image, lors):
        lors = lors.transpose()
        return Tensor(Op.get_module().projection(
            lors=lors.data,
            image=image.data,
            grid=image.grid,
            center=image.center,
            size=image.size,
            tof_bin=self.config(self.KEYS.TOF_BIN),
            tof_sigma2=self.config(self.KEYS.TOF_SIGMA2)))

    def check_inputs(self, data, name):
        if not isinstance(data, dict):
            raise TypeError(
                "{} should be dict, got {}.".format(name, data))

    def backprojection(self, lors, image):
        lors_value = lors['lors_value']
        lors = lors['lors']
        lors = lors.transpose()
        result = Tensor(Op.get_module().backprojection(
            image=image.data,
            grid=image.grid,
            center=image.center,
            size=image.size,
            lors=lors.data,
            lors_value=lors_value.data,
            tof_bin=self.config(self.KEYS.TOF_BIN),
            tof_sigma2=self.config(self.KEYS.TOF_SIGMA2)))
        return result

    def maplors(self, lors, image: Image):
        lors_value = lors['lors_value']
        lors = lors['lors']
        lors = lors.transpose()
        result = Tensor(Op.get_module().maplors(
            image=image.data,
            grid=image.grid,
            center=image.center,
            size=image.size,
            lors=lors.data,
            lors_value=lors_value.data))
        return result
