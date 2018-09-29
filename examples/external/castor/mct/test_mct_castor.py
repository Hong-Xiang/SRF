import os
import numpy as np
import  pytest
from srf.test import TestCase

class TestBBS(TestCase):
    def get_mct_path(self):
        return '/mnt/gluster/Techpi/mct/recon/castor'

    def test_16module_dernezo(self):
        cmd = '. '+self.get_mct_path()+'/run_castor.sh'
        os.system(cmd)
        fortest_result = np.fromfile(self.get_mct_path()+'/cylinder_cas/cylinder_cas_it10.img',dtype='float32')
        new_recon_result = np.fromfile('./cylinder_cas/cylinder_cas_it10.img',dtype='float32')
        assert (fortest_result==new_recon_result).all()

