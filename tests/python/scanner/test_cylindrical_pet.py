
from srf.test import TestCase
import pytest
from srf.scanner.pet.pet import CylindricalPET
from srf.config import clear_config, update_config


class TestCylindicalPET(TestCase):

    def setUp(self):
        clear_config()

    def tearDown(self):
        clear_config()

    def get_mct_default_config(self):
        return {'scanner': {
            "modality": "PET",
            "name": "mCT",
            "ring": {
                "inner_radius": 424.5,
                "outer_radius": 444.5,
                "axial_length": 220.0,
                "nb_rings": 4,
                "nb_blocks_ring": 48,
                "gap": 4.0
            },
            "block": {
                "grid": [13, 13, 1],
                "size": [52.0, 52.0, 20.0],
                "interval": [0.0, 0.0, 0.0]
            },
            "tof": {
                "resolution": 530,
                "bin": 40
            }
        }}

    def get_brain_8_default_config(self):
        return {'scanner': {
            "modality": "PET",
            "name": "brain_8",
            "ring": {
                "inner_radius": 50.0,
                "outer_radius": 70.0,
                "axial_length": 33.4,
                "nb_rings": 1,
                "nb_blocks_ring": 8,
                "gap": 0.0
            },
            "block": {
                "grid": [10, 10, 1],
                "size": [33.4, 33.4, 20.0],
                "interval": [0.0, 0.0, 0.0]
            },
            "tof": {
                "resolution": 100000,
                "bin": 40
            }
        }
        }

    def set_test_cylindicalpet_config(self):
        update_config(self.get_mct_default_config())

    def set_test_single_ring_config(self):
        update_config(self.get_brain_8_default_config())

    def assertConfigEqual(self, scanner, expected_config):

        self.assertEqual(scanner.config('name'),
                         expected_config['name'], 'Invalid scanner name')
        self.assertEqual(scanner.config('inner_radius'),
                         expected_config['ring']['inner_radius'], 'Invalid inner radius')
        self.assertEqual(scanner.config('outer_radius'),
                         expected_config['ring']['outer_radius'], 'Invalid outer radius')
        self.assertEqual(scanner.config('axial_length'),
                         expected_config['ring']['axial_length'], 'Invalid axial length')
        self.assertEqual(scanner.config('nb_rings'),
                         expected_config['ring']['nb_rings'], 'Invalid ring number')
        self.assertEqual(scanner.config('nb_blocks_per_ring'),
                         expected_config['ring']['nb_blocks_per_ring'], 'Invalid block number')
        self.assertEqual(scanner.config('gap'),
                         expected_config['ring']['gap'], 'Invalid ring gap')
        self.assertEqual(scanner.config('grid'),
                         expected_config['block']['grid'], 'Invalid block grid')
        self.assertEqual(scanner.config('size'),
                         expected_config['block']['size'], 'Invalid block size')
        self.assertEqual(scanner.config('interval'),
                         expected_config['block']['interval'], 'Invalid crystal interval')
        self.assertEqual(scanner.config('tof')['resolution'],
                         expected_config['tof']['resolution'], 'Invalid tof resolution')
        self.assertEqual(scanner.config('tof')['bin'],
                         expected_config['tof']['bin'], 'Invalid tof bin size')
    
    @pytest.mark.skip(reason="no way of currently testing this")
    def test_init(self):
        self.set_test_cylindicalpet_config()
        scanner = CylindricalPET()
        self.assertConfigEqual(scanner, self.get_mct_default_config())
    
    @pytest.mark.skip(reason="no way of currently testing this")
    def test_make_rings(self):
        self.set_test_single_ring_config()
        scanner = CylindricalPET()
        result = scanner.make_rings()
        self.assertEqual(len(result), scanner.config('ring')[
                         'nb_rings'], 'Invalid number of rings')
        self.assertEqual(len(result[0], scanner.config('ring')[
                         'nb_blocks_per_ring']), 'Invalid number of blocks per ring')

    @pytest.mark.skip(reason="no way of currently testing this")
    def test_make_block_pairs(self):

        pass
    @pytest.mark.skip(reason="no way of currently testing this")
    def test_make_ring_pairs(self):
        
        pass
    @pytest.mark.skip(reason="no way of currently testing this")
    def test_locate_events(self):
        pass
