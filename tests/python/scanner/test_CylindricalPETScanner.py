from srf import CylindricalPETScanner
from srf import CylindricalPETScannerSpec
import unittest

class TestCylindicalPETScanner(unittest.TestCase):
    def test_construct(self):
        scanner = CylindricalPETScanner(CylindricalPETScannerSpec(nb_rings=100, nb_blocks=10))
        self.assertEqual(scanner.config(scanner.KEYS.CONFIG.NB_RINGS), 100)

    def test_make_rings(self):
        pass
    
    
    def test_make_block_pairs(self):
        pass
    
    def test_make_ring_pairs(self):
        pass

    def test_locate_events(self):
        pass
    


        
        