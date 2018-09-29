import os
import numpy as np
import  pytest
from srf.test import TestCase

class TestBBS(TestCase):
    def get_16module_cylinder_path(self):
        return '/mnt/gluster/Techpi/brain16/recon/lmrec/lm_cylinder'

    def test_16module_dernezo(self):
        cmd = '. '+self.get_16module_cylinder_path()+'/run_lmrec.sh'
        os.system(cmd)
        fortest_result = np.fromfile(self.get_16module_cylinder_path()+'/output_10.rec',np.float32)
        new_recon_result = np.fromfile('output_10.rec',np.float32)
        assert (fortest_result==new_recon_result).all()

