from srf.test import TestCase
import pytest
from dxl.learn.core import Model


class WorkerGraphTestCase(TestCase):
    def dummpy_recon_step_maker(self, inputs):
        class ReconStep(Model):
            pass

        def maker(inputs):
            pass

        return maker


class TestWorkerGraph(TestCase):
    def get_graph(self):
        pass

    def test_x_linked(self):
        pass

    def test_x_target_linked(self):
        pass

    def test_recon_model_input_linked(self):
        pass

    def test_recon_model_output_linked_with_target(self):
        pass

    def test_run_updated_target(self):
        pass


@pytest.mark.skip('not impl yet')
class TestOSEMWorkerGraph(TestCase):
    def test_subset_linked(self):
        master = self.make_master_graph()
        worker = self.make_graph(master)
        for i in range(mg.nb_workers):
            assert master.tensor('subset') is worker.tensor('subset')
