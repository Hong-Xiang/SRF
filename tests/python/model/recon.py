from dxl.learn.core import Model, Tensor


# from dxl.learn.model.tor_recon import Projection, BackProjection
import tensorflow as tf
import numpy as np
import warnings
import os
TF_ROOT = os.environ.get('TENSORFLOW_ROOT')
op = tf.load_op_library(
    TF_ROOT + '/bazel-bin/tensorflow/core/user_ops/tof_tor.so')
warnings.warn(DeprecationWarning())

projection = op.projection_gpu
backprojection = op.backprojection_gpu


def Projection(name, image,
               grid, center, size,
               lors,
               tof_bin, tof_sigma2,
               kernel_width):
    grid = np.array(grid, dtype=np.int32)
    center = np.array(center, dtype=np.float32)
    size = np.array(size, dtype=np.float32)
    kernel_width = float(kernel_width)
    tof_bin = float(tof_bin)
    tof_sigma2 = float(tof_sigma2)
    print(tof_bin)

    img = tf.constant(image, tf.float32)
    lors = tf.constant(lors, tf.float32)
    lors = tf.transpose(lors)

    projection_value = projection(
        lors=lors,
        image=img,
        grid=grid,
        center=center,
        size=size,
        kernel_width=kernel_width,
        tof_bin=tof_bin,
        tof_sigma2=tof_sigma2)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        result = sess.run(projection_value)
    tf.reset_default_graph()
    return projection_value


def BackProjection(name, image,
                   grid, center, size,
                   lors, lor_value,
                   tof_bin, tof_sigma2,
                   kernel_width):

        grid = np.array(grid, dtype=np.int32)
        center = np.array(center, dtype=np.float32)
        size = np.array(size, dtype=np.float32)
        kernel_width = float(kernel_width)
        tof_bin = float(tof_bin)
        tof_sigma2 = float(tof_sigma2)

        img = tf.constant(image, tf.float32)
        lors = tf.constant(lors, tf.float32)
        lors = tf.transpose(lors)
        lor_value = tf.constant(lors, tf.float32)
        lor_value = tf.transpose(lors)

        backprojection_image = backprojection(
            lors=lors,
            image=img,
            grid=grid,
            center=center,
            size=size,
            lor_values=py,
            kernel_width=kernel_width,
            tof_bin=tof_bin,
            tof_sigma2=tof_sigma2)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            result = sess.run(backprojection_image)
        tf.reset_default_graph()
        return backprojection_image