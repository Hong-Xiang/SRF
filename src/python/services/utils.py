import numpy as np

import tensorflow as tf
import json
import logging

from dxl.learn.core import ThisHost


# def ensure_float32(x):
#     if isinstance(x, np.ndarray) and x.dtype == np.float64:
#         return np.array(x, dtype=np.float32)
#     return x


# def constant_tensor(x, name, ginfo):
#   return TensorNumpyNDArray(_ensure_float32(x), None, ginfo.update(name=name))


# def variable_tensor(x, name, ginfo):
#   x_tensor = TensorVariable(
#       VariableInfo(None, _ensure_float32(x), tf.float32),
#       ginfo.update(name=name))
#   x_init = x_tensor.assign(_constant_tensor(x, name + '_initial_value', ginfo))
#   return x_tensor, x_init


def print_tensor(t, name=None):
    print("[DEBUG] name: {}, tensor: {}, value:\n{}".format(
        name, t.data, t.run()))


def debug_tensor(t, msg, logger):
    logger.debug("Debug {}, tensor: {}, (.data: {}):\n{}".format(
        msg, t, t.data, t.run()))


def print_info(*msg):
    print('INFO', *msg)