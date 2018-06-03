import numpy as np
import pytest

from srf.graph.reconstruction.osem import OSEMMasterGraph
from srf.test import TestCase


class TestOSEMMasterGraph(TestCase):
    def get_graph(self):
        x = np.ones([5] * 3)
        return OSEMMasterGraph('master', initial_image=x, nb_workers=2, nb_subsets=10)

    def test_subset_increase(self):
        g = self.get_graph()
        with self.variables_initialized_test_session() as sess:
            assert sess.run(g.tensor(g.KEYS.TENSOR.SUBSET)) == 0
            for i in range(20):
                sess.run(g.tensor(g.KEYS.TENSOR.INC_SUBSET))
                assert sess.run(g.tensor(g.KEYS.TENSOR.SUBSET)
                                ) == (i + 1) % g.nb_subsets

    def test_update_depens_on_subset_inc(self):
        g = self.get_graph()
        with self.variables_initialized_test_session() as sess:
            assert sess.run(g.tensor(g.KEYS.TENSOR.SUBSET)) == 0
            for i in range(20):
                sess.run(g.tensor(g.KEYS.TENSOR.UPDATE))
                assert sess.run(g.tensor(g.KEYS.TENSOR.SUBSET)
                                ) == (i + 1) % g.nb_subsets


@pytest.mark.skip('not impl yet')
class TestOSEMWorkerGraph(TestCase):
    def test_subset_linked(self):
        master = self.make_master_graph()
        worker = self.make_graph(master)
        for i in range(mg.nb_workers):
            assert master.tensor('subset') is worker.tensor('subset')
