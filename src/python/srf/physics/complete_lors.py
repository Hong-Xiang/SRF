import tensorflow as tf
import os
from dxl.learn.tensor import transpose
from srf.model import projection, backprojection
from srf.data import Image, ListModeData, ListModeDataWithoutTOF
from srf.utils.config import config_with_name

# load op
TF_ROOT = os.environ.get('TENSORFLOW_ROOT')

op_dir = '/bazel-bin/tensorflow/core/user_ops/'
# op_list = {
#     'siddon': 'siddon.so',
#     'tor': 'tof_tor.so'
# }


class Op:
    op = None

    @classmethod
    def load(cls):
        cls.op = tf.load_op_library(
            TF_ROOT + op_dir + 'siddon.so')

    @classmethod
    def get_module(cls):
        if cls.op is None:
            cls.load()
        return cls.op


class CompleteLoRsModel:
    """
    This model provides support to the models (typically for siddon model)
    using complete lors.This model processes the lors dataset without splitting.
    """

    class KEYS:
        class CONFIG:
            TOF_SIGMA2 = 'tof_sigma2'
            TOF_BIN = 'tof_bin'

    def __init__(self, name, tof_sigma2=None, tof_bin=None):
        self.name = name
        self.config = config_with_name(name)
        if tof_sigma2 is None:
            tof_sigma2 = 1e10
        if tof_bin is None:
            tof_bin = 1e10
        self.config.update(self.KEYS.CONFIG.TOF_SIGMA2, tof_sigma2)
        self.config.update(self.KEYS.CONFIG.TOF_BIN, tof_bin)

    @property
    def op(self):
        return Op.get_module()


@projection.register(CompleteLoRsModel, Image, ListModeData)
def _(physical_model, image, projection_data):
    image = transpose(image)
    result = physical_model.op.projection(
        lors=transpose(projection_data.lors),
        image=image.data,
        grid=list(image.grid[::-1]),
        center=list(image.center[::-1]),
        size=list(image.size[::-1]),
        tof_bin=physical_model.config[physical_model.KEYS.CONFIG.TOF_BIN],
        tof_sigma2=physical_model.config[physical_model.KEYS.CONFIG.TOF_SIGMA2])
    return ListModeData(projection_data.lors, result)


@backprojection.register(CompleteLoRsModel, ListModeData, Image)
def _(physical_model, projection_data, image):
    image = transpose(image)
    print(transpose(projection_data.lors))
    result = physical_model.op.backprojection(
        image=image.data,
        grid=list(image.grid[::-1]),
        center=list(image.center[::-1]),
        size=list(image.size[::-1]),
        lors=transpose(projection_data.lors),
        lors_value=projection_data.values,
        tof_bin=physical_model.config[physical_model.KEYS.CONFIG.TOF_BIN],
        tof_sigma2=physical_model.config[physical_model.KEYS.CONFIG.TOF_SIGMA2])
    return transpose(Image(result, image.center[::-1], image.size[::-1]))


@backprojection.register(CompleteLoRsModel, ListModeDataWithoutTOF, Image)
def _(physical_model, projection_data, image):
    image = transpose(image)
    result = physical_model.op.maplors(
        image=image.data,
        grid=list(image.grid[::-1]),
        center=list(image.center[::-1]),
        size=list(image.size[::-1]),
        lors=transpose(projection_data.lors),
        lors_value=projection_data.values)
    return transpose(Image(result, image.center[::-1], image.size[::-1]))
