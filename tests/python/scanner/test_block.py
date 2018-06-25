import numpy as np
from srf.test import TestCase
import pytest

from srf.scanner.pet.block import RingBlock
from srf.scanner.geometry import Vec3 

class BlockTestBase(TestCase):

    def setUp(self):
        super().setUp()

    def get_attrs(self):
        grid = [1, 1, 1]
        size = [ 6.0, 2.0, 1.0]
        center = [1.0 ,0.0 ,0.0]
        rad_z = np.pi/2
        return [ size, center,grid, rad_z]
    
    def get_meshes(self):
        return np.array([[0.0, 1.0, 0.0],])


class TestRingBlock(BlockTestBase):

    def make_RingBlock(self):
        attrs = self.get_attrs()
        size = attrs[0]
        center = attrs[1]
        grid = attrs[2]
        rad_z = attrs[3]
        return RingBlock( size, center, grid, rad_z)

    def test_make_block(self):
        b = self.make_RingBlock()
        assert isinstance(b , RingBlock)
    
    def test_make_meshes(self):
        b = self.make_RingBlock()
        result_meshes = b.get_meshes()
        expected_meshes = self.get_meshes()
        self.assertFloatArrayEqual(result_meshes, expected_meshes)
