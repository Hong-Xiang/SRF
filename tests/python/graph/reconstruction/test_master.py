from srf.test import TestCase
from srf.graph.reconstruction.master import MasterGraph, OSEMMasterGraph
from dxl.learn.core import Variable
import numpy as np


class TestMasterGraph(TestCase):
    def get_graph(self, x=None, nb_workers=None):
        if x is None:
            x = np.ones([5, 5, 5])
        if nb_workers is None:
            nb_workers = 2

        class DummyLoader:
            def __init__(self, config):
                pass

            def load(self):
                return x
        return MasterGraph('master', local_loader_cls=DummyLoader, nb_workers=nb_workers)

    def test_image_init(self):
        x_init = np.ones([10] * 3)
        g = self.get_graph(x_init)
        with self.test_session() as sess:
            sess.run(g.tensor(g.KEYS.TENSOR.INIT))
            xv = sess.run(g.tensor(g.KEYS.TENSOR.X))
        self.assertFloatArrayEqual(xv, x_init)

    def test_buffers(self):
        g = self.get_graph()
        x = g.tensor(g.KEYS.TENSOR.X)
        for i in range(g.nb_workers):
            b = g.tensor(g.KEYS.TENSOR.BUFFER)[i]
            assert isinstance(b, Variable)
            assert b.shape == x.shape
            assert b.dtype == x.dtype
            assert not b.data == x.data

    def test_merge(self):
        g = self.get_graph()
        for i in range(g.nb_workers):
            assert g.subgraph('summation').tensor('input')[
                i] is g.tensor(g.KEYS.TENSOR.BUFFER)[i]

    def assign_buffers_values(self, g):
        return 3 * np.ones(g.tensor(g.KEYS.TENSOR.X).shape)

    def assign_buffers_ops(self, g):
        assigns = []
        for i in range(g.nb_workers):
            assigns.append(g.tensors[g.KEYS.TENSOR.BUFFER]
                           [i].assign(self.assign_buffers_values(g)))
        return assigns

    def test_assign_buffers_ops(self):
        g = self.get_graph()
        with self.variables_initialized_test_session() as sess:
            assigns = self.assign_buffers_ops(g)
            for a in assigns:
                sess.run(a)
            for i in range(g.nb_workers):
                self.assertFloatArrayEqual(sess.run(g.tensor(g.KEYS.TENSOR.BUFFER)[
                    i]), self.assign_buffers_values(g))

    def test_merge_run(self):
        g = self.get_graph()
        with self.variables_initialized_test_session() as sess:
            assigns = self.assign_buffers_ops(g)
            for a in assigns:
                sess.run(a)
            sess.run(g.tensor(g.KEYS.TENSOR.UPDATE))
            self.assertFloatArrayEqual(sess.run(g.tensor(g.KEYS.TENSOR.X)),
                                       self.assign_buffers_values(g) * g.nb_workers)

    def test_init_op(self):
        g = self.get_graph()
        KT = g.KEYS.TENSOR
        with self.test_session() as sess:
            sess.run(g.tensor(KT.INIT))
            sess.run(g.tensor(KT.X))
            sess.run(g.tensor(KT.BUFFER))


class TestOSEMMasterGraph(TestCase):
    def get_graph(self):
        x = np.ones([5] * 3)

        class DummyLoader:
            def __init__(self, config):
                pass

            def load(self):
                return np.ones([5] * 3)
        return OSEMMasterGraph('master', local_loader_cls=DummyLoader, nb_workers=2, nb_subsets=10)

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

    def test_init_op(self):
        g = self.get_graph()
        KT = g.KEYS.TENSOR
        with self.test_session() as sess:
            sess.run(g.tensor(KT.INIT))
            sess.run(g.tensor(KT.X))
            sess.run(g.tensor(KT.BUFFER))
            sess.run(g.tensor(KT.SUBSET))
