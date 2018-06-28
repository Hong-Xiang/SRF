import numpy as np
from srf.test import TestCase
import pytest

from srf.scanner.pet.block import BlockPair, RingBlock
# from srf.scanner.geometry import Vec3 

class BlockTestBase(TestCase):

    def setUp(self):
        super().setUp()

    def get_attrs(self):
        size = np.array([ 6.0, 2.0, 1.0])
        grid = np.array([1, 1, 1])
        center = np.array([1.0 ,0.0 ,0.0])
        rad_z = np.pi/2
        return size, grid, center, rad_z
    
    def get_meshes(self):
        return np.array([[0.0, 1.0, 0.0],])

class TestRingBlock(BlockTestBase):

    def make_RingBlock(self):
        size, grid, center, rad_z = self.get_attrs()
        return RingBlock( size, grid, center, rad_z)

    def compare_block(self,rb, eb):
        self.assertFloatArrayEqual(rb.block_size,eb[0])
        self.assertFloatArrayEqual(rb.grid, eb[1])
        self.assertFloatArrayEqual(rb.center, eb[2])
        self.assertFloatArrayEqual(rb.rad_z, eb[3])    

    def test_make_block(self):
        result_block = self.make_RingBlock()
        expected_block = self.get_attrs()
        assert isinstance(result_block, RingBlock)
        self.compare_block(result_block, expected_block)
    
    def test_make_meshes(self):
        b = self.make_RingBlock()
        result_meshes = b.get_meshes()
        expected_meshes = self.get_meshes()
        self.assertFloatArrayEqual(result_meshes, expected_meshes)

class TestBlockPair(BlockTestBase):
    
    def make_block_pair(self):
        size, grid, center, rad_z = self.get_attrs()
        block1 = RingBlock(size, grid, center, rad_z = 0)
        block2 = RingBlock(size, grid, center, rad_z = np.pi)
        return BlockPair(block1, block2)

    def test_make_block_pair(self):
        bp = self.make_block_pair()
        assert isinstance(bp, BlockPair)

    def test_make_lors(self):
        block_pair = self.make_block_pair()
        result_lors = block_pair.make_lors()
        expect_lors = [[1.0, 0.0, 0.0, -1.0, 0.0, 0.0], ]
        self.assertFloatArrayEqual(result_lors, expect_lors)