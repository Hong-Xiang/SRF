import pytest
import numpy as np
from srf.test import TestCase

from srf.scanner.pet.pet import CylindricalPET

class ScannerTestBase(TestCase):

    def setUp(self):
        super().setUp()
    

class TestCylindricalPET(TestCase):

    def test_make_rings(self):
    
    @pytest.mark.skip(reason='')
    def test_map_lors(self):
