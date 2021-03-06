import tensorflow as tf
from doufo.tensor import sum_
from srf.model import projection, backprojection
from srf.utils.config import config_with_name
from srf.data import ListModeDataSplit, ListModeDataSplitWithoutTOF, Image, ListModeData
from dxl.learn.tensor import transpose
import os

# load op
TF_ROOT = os.environ.get('TENSORFLOW_ROOT')

op_dir = '/bazel-bin/tensorflow/core/user_ops/'
class Op:
    op = None

    @classmethod
    def load(cls):
        cls.op = tf.load_op_library(
            TF_ROOT + op_dir + 'tor.so')

    @classmethod
    def get_module(cls):
        if cls.op is None:
            cls.load()
        return cls.op


class SplitLoRsModel:
    """
    This model provides support to the models (typically for tor model) using split lors.
    In these model, lors are split into group and processed respectively.
    """

    AXIS = ('x', 'y', 'z')

    class KEYS:
        KERNEL_WIDTH = 'kernel_width'
        TOF_BIN = 'tof_bin'
        TOF_SIGMA2 = 'tof_sigma2'
        GEO_SIGMA2_FLAG = 'geo_sigma2_flag'

    def __init__(self, kernel_width, geo_sigma2_flag=False, tof_sigma2=None, tof_bin=None, name='split_lor_model'):
        if tof_bin is None:
            tof_bin = 1.0e4
        if tof_sigma2 is None:
            tof_sigma2 = 1.0e4
        self.config = config_with_name(name)
        self.config.update(self.KEYS.KERNEL_WIDTH, kernel_width)
        # print("debug, ", geo_sigma2_flag)
        self.config.update(self.KEYS.GEO_SIGMA2_FLAG, geo_sigma2_flag)
        self.config.update(self.KEYS.TOF_SIGMA2, tof_sigma2)
        self.config.update(self.KEYS.TOF_BIN, tof_bin)

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


@projection.register(SplitLoRsModel, Image, ListModeDataSplit)
def _(model, image, projection_data):
    result = {}
    # print("tof_sigma2: ", model.config[model.KEYS.TOF_SIGMA2])
    for a in model.AXIS:
    # for a in ('x','y'):
        image_axis = transpose(image, model.perm(a))
        result[a] = Op.get_module().projection(
            lors=transpose(projection_data[a].lors),
            image=image_axis.data,
            grid=list(image_axis.grid[::-1]),
            center=list(image_axis.center[::-1]),
            size=list(image_axis.size[::-1]),
            kernel_width=model.config[model.KEYS.KERNEL_WIDTH],
            geo_sigma2_flag=model.config[model.KEYS.GEO_SIGMA2_FLAG],
            tof_bin=model.config[model.KEYS.TOF_BIN],
            tof_sigma2=model.config[model.KEYS.TOF_SIGMA2])
    return ListModeDataSplit(*[ListModeData(projection_data[a].lors, result[a]) for a in model.AXIS])


@backprojection.register(SplitLoRsModel, ListModeDataSplit, Image)
def _(model, projection_data, image):
    result = []
    for a in model.AXIS:
    # for a in ['x','y']:
        image_axis = transpose(image, model.perm(a))
        # print(transpose(projection_data[a].lors))
        # print(projection_data[a].lors)
        backproj = Op.get_module().backprojection(
            image=image_axis.data,
            grid=list(image_axis.grid[::-1]),
            center=list(image_axis.center[::-1]),
            size=list(image_axis.size[::-1]),
            lors=transpose(projection_data[a].lors),
            lors_value=projection_data[a].values,
            kernel_width=model.config[model.KEYS.KERNEL_WIDTH],
            geo_sigma2_flag=model.config[model.KEYS.GEO_SIGMA2_FLAG],
            tof_bin=model.config[model.KEYS.TOF_BIN],
            tof_sigma2=model.config[model.KEYS.TOF_SIGMA2])
        result.append(transpose(backproj, model.perm_back(a)))
    return Image(sum_(result), image.center, image.size)


@backprojection.register(SplitLoRsModel, ListModeDataSplitWithoutTOF, Image)
def _(model, projection_data, image):
    result = []
    # print("geometry sigma2 flag:",model.config[model.KEYS.GEO_SIGMA2_FLAG])
    for a in model.AXIS:
    # for  a in ['x', 'z']:
        image_axis = transpose(image, model.perm(a))
        backproj = Op.get_module().maplors(
            image=image_axis.data,
            grid=list(image_axis.grid[::-1]),
            center=list(image_axis.center[::-1]),
            size=list(image_axis.size[::-1]),
            lors=transpose(projection_data[a].lors),
            lors_value=projection_data[a].values,
            kernel_width=model.config[model.KEYS.KERNEL_WIDTH],
            geo_sigma2_flag=model.config[model.KEYS.GEO_SIGMA2_FLAG])
        result.append(transpose(backproj, model.perm_back(a)))
    return Image(sum_(result), image.center, image.size)
