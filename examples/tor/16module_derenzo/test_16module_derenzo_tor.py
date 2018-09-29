import os
import numpy as np
import  pytest
from srf.test import TestCase

class TestBBS(TestCase):
    def get_16module_derenzo_path(self):
        return '/mnt/gluster/Techpi/brain16/recon/SRF_tor/derenzo'

    def test_16module_dernezo(self):
        cmd = '. '+self.get_16module_derenzo_path()+'/run_tor.sh'
        os.system(cmd)
        fortest_result = np.load(self.get_16module_derenzo_path()+'/recon_10.npy')
        new_recon_result = np.load('recon_10.npy')
        assert (fortest_result==new_recon_result).all()

