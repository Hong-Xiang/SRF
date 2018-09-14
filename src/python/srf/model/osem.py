from dxl.learn import Model
from srf.utils.config import config_with_name
import tensorflow as tf


class OrderedSubset(Model):
    class KEYS:
        class CONFIG:
            NB_SUBSETS = 'nb_subsets'

    def __init__(self, nb_subsets, name):
        self.config = config_with_name(name)
        self.config.update(self.KEYS.CONFIG.NB_SUBSETS, nb_subsets)

    def build(self, x):
        if isinstance(x, (tf.Tensor, tf.Variable, tf.Operation)):
            # TODO add safer implementation
            with tf.variable_scope(None, default_name="osem_step"):
                self.step = tf.get_variable("step")
        raise TypeError(f"OrderedSubset does not support {type(x)}")

    def kernel(self):
        pass

    def parameters(self):
        return []
