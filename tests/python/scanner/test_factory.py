import unittest
from srf.scanner import ScannerFactory
from srf.scanner.pet import PETScanner, CylindricalPETScanner, PatchPETScanner
from srf.config import clear_config, update_config


class TestScannerFactory(unittest.TestCase):
    def setUp(self):
        clear_config()

    def tearDown(self):
        clear_config()

    def set_test_scanner_configs(self):
        update_config({'scanner': {
            'modality': 'PET',
            'name': 'mCT',
            "ring": {
                "inner_radius": 424.5,
                "outer_radius": 444.5,
                "axial_length": 220.0,
                "nb_ring": 4,
                "nb_block_ring": 48,
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
        }
        })
    

    def test_make_scanner(self):
        self.set_test_scanner_configs()
        scanner_factory = ScannerFactory()
        scanner = scanner_factory.make_scanner()
        self.assertIsInstance(scanner, CylindricalPETScanner, 'Wrong Scanner class.')
    