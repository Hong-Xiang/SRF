import tensorflow as tf
import os
from dxl.learn.tensor import transpose
from srf.model import projection, backprojection
from srf.data import Image, SinogramData
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
            TF_ROOT + op_dir + 'siddon_sino.so')

    @classmethod
    def get_module(cls):
        if cls.op is None:
            cls.load()
        return cls.op


class CompleteSinoModel:
    """
    This model provides support to the models (typically for siddon model)
    using complete lors.This model processes the lors dataset without splitting.
    """

    class KEYS:
        class CONFIG:
            BLOCK_GRID = 'block_grid'
            BLOCK_SIZE = 'block_size'
            BLOCK_CENTER = 'block_center'
            INNER_RADIUS = 'inner_radius'
            OUTER_RADIUS = 'outer_radius'
            NB_RINGS = 'nb_rings'
            NB_BLOCKS_PER_RING = 'nb_blocks_per_ring'
            GAP = 'gap'
    def __init__(self, name, pj_config):
        self.name = name
        self.config = config_with_name(name)
        # self.config.update(self.KEYS.CONFIG.BLOCK_GRID, [1, 10, 10])
        # self.config.update(self.KEYS.CONFIG.BLOCK_SIZE, [20.0, 33.4, 33.4])
        # self.config.update(self.KEYS.CONFIG.BLOCK_CENTER, [0.0, 0.0, 0.0])
        # self.config.update(self.KEYS.CONFIG.INNER_RADIUS, 99.0)
        # self.config.update(self.KEYS.CONFIG.OUTER_RADIUS, 119.0)
        # self.config.update(self.KEYS.CONFIG.NB_RINGS, 1)
        # self.config.update(self.KEYS.CONFIG.NB_BLOCKS_PER_RING, 16)
        # self.config.update(self.KEYS.CONFIG.GAP, 0.0)
        self.config.update(self.KEYS.CONFIG.BLOCK_GRID, pj_config['block']['grid'])
        self.config.update(self.KEYS.CONFIG.BLOCK_SIZE, pj_config['block']['size'])
        self.config.update(self.KEYS.CONFIG.BLOCK_CENTER, pj_config['block']['center'])
        self.config.update(self.KEYS.CONFIG.INNER_RADIUS, pj_config['ring']['inner_radius'])
        self.config.update(self.KEYS.CONFIG.OUTER_RADIUS, pj_config['ring']['outer_radius'])
        self.config.update(self.KEYS.CONFIG.NB_RINGS, pj_config['ring']['nb_rings'])
        self.config.update(self.KEYS.CONFIG.NB_BLOCKS_PER_RING, pj_config['ring']['nb_blocks_per_ring'])
        self.config.update(self.KEYS.CONFIG.GAP, pj_config['ring']['gap'])

    @property
    def op(self):
        return Op.get_module()


@projection.register(CompleteSinoModel, Image, SinogramData)
def _(physical_model, image, projection_data):
    image = transpose(image)
    result = physical_model.op.projection(
        sino=projection_data.data,
        image=image.data,
        grid=list(image.grid[::-1]),
        center=list(image.center[::-1]),
        size=list(image.size[::-1]),
        block_grid  = list(physical_model.config[physical_model.KEYS.CONFIG.BLOCK_GRID][::-1]),
        block_size = list(physical_model.config[physical_model.KEYS.CONFIG.BLOCK_SIZE][::-1]),
        block_center = list(physical_model.config[physical_model.KEYS.CONFIG.BLOCK_CENTER][::-1]),
        inner_radius = physical_model.config[physical_model.KEYS.CONFIG.INNER_RADIUS],
        outer_radius = physical_model.config[physical_model.KEYS.CONFIG.OUTER_RADIUS],
        nb_rings = physical_model.config[physical_model.KEYS.CONFIG.NB_RINGS],
        nb_blocks_per_ring = physical_model.config[physical_model.KEYS.CONFIG.NB_BLOCKS_PER_RING],
        gap = physical_model.config[physical_model.KEYS.CONFIG.GAP])
    return SinogramData(result)


@backprojection.register(CompleteSinoModel, SinogramData, Image)
def _(physical_model, projection_data, image):
    image = transpose(image)
    result = physical_model.op.backprojection(
        image=image.data,
        sino=projection_data.data,
        grid=list(image.grid[::-1]),
        center=list(image.center[::-1]),
        size=list(image.size[::-1]),
        block_grid  = list(physical_model.config[physical_model.KEYS.CONFIG.BLOCK_GRID][::-1]),
        block_size = list(physical_model.config[physical_model.KEYS.CONFIG.BLOCK_SIZE][::-1]),
        block_center = list(physical_model.config[physical_model.KEYS.CONFIG.BLOCK_CENTER][::-1]),
        inner_radius = physical_model.config[physical_model.KEYS.CONFIG.INNER_RADIUS],
        outer_radius = physical_model.config[physical_model.KEYS.CONFIG.OUTER_RADIUS],
        nb_rings = physical_model.config[physical_model.KEYS.CONFIG.NB_RINGS],
        nb_blocks_per_ring = physical_model.config[physical_model.KEYS.CONFIG.NB_BLOCKS_PER_RING],
        gap = physical_model.config[physical_model.KEYS.CONFIG.GAP])
    return transpose(Image(result, image.center[::-1], image.size[::-1]))

