import tensorflow as tf
from dxl.learn.core import ConfigurableWithName
# load op
TF_ROOT = os.environ.get('TENSORFLOW_ROOT')
op = tf.load_op_library(
    TF_ROOT + '/bazel-bin/tensorflow/core/user_ops/tof_tor.so')


class ToRModel(ConfigurableWithName):
    class KEYS:
        KERNEL_WIDTH = 'kernel_width'
        TOF_BIN = 'tof_bin'
        TOF_SIGMA2 = 'tof_sigma2'

    def __init__(self, name, *, config, kernel_width, tof_bin, tof_sigma2):
        self._parse_input_config(config, {
            self.KEYS.KERNEL_WIDTH: kernel_width,
            self.KEYS.TOF_BIN: tof_bin,
            self.KEYS.TOF_SIGMA2: tof_sigma2
        })

    @classmethod
    def _default_config(self):
        return {
            self.KEYS.KERNEL_WIDTH: 1.0,
            self.KEYS.TOF_BIN: 1.0,
            self.KEYS.TOF_SIGMA2: 1.0,
        }
    AXIS = ('x', 'y', 'z')

    def perm(self, axis):
        if axis == 'z':
            return [2, 1, 0]
        if axis == 'y':
            return [1, 2, 0]
        if axis == 'x':
            return [0, 2, 1]

    def rotate_param(self, value, axis):
        return [value[p] for p in self.perm(a)]

    def projection(self, lors, image):
        return op.projection_gpu(
            lors=lors, image=image,
            grid=image.grid,
            center=image.center,
            size=image.size,
            kernel_width=self.config(self.KEYS.KERNEL_WIDTH),
            tof_bin=self.config(self.KEYS.TOF_BIN),
            tof_sigma2=self.config(self.KEYS.TOF_SIGMA2))

    def check_inputs(self, data, name):
        if not isinstance(data, dict):
            raise TypeError(
                "{} should be dict, got {}.".format(name, data))
        for a in self.projection_model.AXIS:
            if not a in data:
                raise ValueError("{} missing axis {}.".format(name, a))

    def backprojection(self, image, lors):
        return op.backprojection_gpu(
            image=image,
            grid=image.grid,
            center=image.center,
            size=image.size,
            lors=lors['lors']
            lor_values=lors['value'],
            kernel_width=self.config(self.KEYS.KERNEL_WIDTH),
            tof_bin=self.config(self.KEYS.TOF_BIN),
            tof_sigma2=self.config(self.KEYS.TOF_SIGMA2))
