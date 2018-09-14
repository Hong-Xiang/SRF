import tensorflow as tf
import unittest
import numpy as np
from dxl.data.io import load_npz

from srf.tensor import LORs


class TestLORs(tf.test.TestCase):
    def create_constant_lors(self):
        lors = np.load('some_path.npy')
        return LORs(lors, name='LORs')

    def create_lors_and_subsets(self):
        lors = np.load('some_path.npy')
        lors_xyz = load_npz('some_path.npz')
        nb_subsets = lors_xyz['nb_subsets']
        lors_xyz = {a: lors_xyz[a] for a in ['x', 'y', 'z']}
        return Lors(lors, name='LORs'), lors_xyz, nb_subsets

    def create_lors_and_backprojection_data(self):
        lors = np.load('some_path.npy')
        back_proj_data = np.load('some_path.npy')

    def test_split_subset(self):
        lors, expect, nb_subsets = self.create_lors_and_subsets()
        lors_xyz = lors.split_subsets(nb_subsets)
        with tf.test_session() as sess:
            result = sess.run(lors_xyz)
        for a in ['x', 'y', 'z']:
            np.test.assertEqual(expect[a], result[a])

    def test_backprojection(self):
        from srf.model import BackProjection
        from srf.physics import Siddon
        # TOR -> linkto op(load from .so)
        lors = self.create_lors_and_backprojection_data()


class TestLORsInXYZ(tf.test.TestCase):
    def test_backprojection(self):
        from srf.model import BackProjection
        from srf.physics import TOR
        lors, expect = self.create_lors_and_backprojection_data()
        bp = lors.backprojection(TOR())
        with tf.test_session() as sess:
            result = sess.run(bp)
        np.test.assertEqual(expect, result)
