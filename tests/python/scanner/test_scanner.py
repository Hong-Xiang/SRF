import pytest
import numpy as np
from srf.test import TestCase

from srf.scanner.pet.block import Block, RingBlock
from srf.scanner.pet.pet import CylindricalPET, MultiPatchPET
from srf.scanner.pet.geometry import RingGeometry
from srf.scanner.pet.spec import TOF

class ScannerTestBase(TestCase):

    def setUp(self):
        super().setUp()


class TestCylindricalPET(ScannerTestBase):

    def get_mct_config(self):
        config = {
            "modality": "PET",
            "name": "mCT",
            "ring": {
                    "inner_radius": 424.5,
                    "outer_radius": 444.5,
                    "axial_length": 220.0,
                    "nb_rings": 4,
                    "nb_blocks_per_ring": 48,
                    "gap": 4.0
            },
            "block": {
                "grid": [1, 13, 13],
                "size": [20.0, 52.0, 52.0],
                "interval": [0.0, 0.0, 0.0]
            },
            "tof": {
                "resolution": 530,
                "bin": 40
            }
        }
        return config
    
    def get_scanner(self):
        config = self.get_mct_config()
        ring = RingGeometry(config['ring'])
        block = Block(block_size = config['block']['size'],
                      grid = config['block']['grid'])
        name = config['name']
        # modality = config['modality']
        tof = TOF(res = config['tof']['resolution'], bin = config['tof']['bin'])
        return CylindricalPET(name, ring, block, tof)

    def get_first_block(self):
        config = self.get_mct_config()
        block_para = config['block']
        ring_para = config['ring']
        z = -(ring_para['gap'] + block_para['size'][2]) * (ring_para['nb_rings']-1) / 2
        x = (ring_para['inner_radius'] + ring_para['outer_radius']) / 2
        pos = [x, 0, z]
        ring_block = RingBlock(block_para['size'], block_para['grid'], pos, 0)
        return ring_block

    def test_make_scanner(self):
        scanner = self.get_scanner()
        assert isinstance(scanner, CylindricalPET)

        result_nb_rings = len(scanner.rings)
        expected_nb_rings = scanner.nb_rings
        self.assertEquals(result_nb_rings, expected_nb_rings)

        result_first_block = scanner.rings[0][0]
        expect_first_block = self.get_first_block()

        self.assertFloatArrayEqual(result_first_block.center, expect_first_block.center)


    @pytest.mark.skip(reason="NIY")
    def test_map_lors(self):
        pass


    def get_dummy_ring(self):
        ring = [1,2,3]
        return ring

    def test_make_block_pairs(self):

        ring1 = self.get_dummy_ring()
        ring2 = self.get_dummy_ring()

        scanner = self.get_scanner()
        block_pairs = scanner.make_block_pairs(ring1, ring2)
        result_block_pairs = [[bp.block1, bp.block2] for bp in block_pairs]
        expect_block_pairs = [[1, 2], [1, 3], [2, 1], [2, 3], [3, 1], [3, 2]]
        self.assertListEqual(result_block_pairs, expect_block_pairs)

        ring3 = ring1
        block_pairs = scanner.make_block_pairs(ring1, ring3)
        result_block_pairs = [[bp.block1, bp.block2] for bp in block_pairs]
        expect_block_pairs = [[1, 2], [1, 3], [2, 3]]
        self.assertListEqual(result_block_pairs, expect_block_pairs)

    @pytest.mark.skip(reason= "NIY")
    def test_make_ring_pet_lors(self):
        pass
class TestMultiPatchPET(ScannerTestBase):
    pass
