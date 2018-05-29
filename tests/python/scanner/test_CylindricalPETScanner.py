from srf import CylindricalPETScanner
from srf import CylindricalPETScannerSpec
import unittest


class TestCylindicalPETScanner(unittest.TestCase):
    def test_construct(self):
        scanner = CylindricalPETScanner(
            CylindricalPETScannerSpec(nb_rings=100, nb_blocks=10))
        self.assertEqual(scanner.config(scanner.KEYS.CONFIG.NB_RINGS), 100)

    def test_make_rings(self):
        pass

    def test_make_block_pairs(self):
        pass

    def test_make_ring_pairs(self):
        pass

    def test_locate_events(self):
        pass

    def test_special(self):

        class TrueLoader:
            def load(file):
                pass

        class DummpyLoader:
            def load(file):
                return 'xxx'

        class Saver:
            pass

        class ServiceManager:
            def get_service(self, key):
                pass

        class DummyServiceManager:
            def get_service(self, key):
                pass

        class FSManager:
            def __init__(self, service_manager):
                # self.loader = loader
                self.loader = ServiceManager.get_service('loader')
                self.saver = ServiceManager.get_service('saver')

            def save(file):
                pass

            def load(file):
                # slow system call
                # with open(file, 'r') as fin:
                #    return fin.read()
                return self.loader.load(file)

        # FSLoader.load = MockObect(return='xxx')
        # FSLoader.load()
        assert FSLoader(DummpyLoader()).load() == 'xxx'
