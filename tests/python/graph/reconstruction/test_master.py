from srf.test import TestCase
from srf.graph.reconstruction.master import MasterGraph
from dxl.learn.core import Variable
import numpy as np


class TestMasterGraph(TestCase):
    def get_graph(self, x=None, nb_workers=None):
        if x is None:
            x = np.ones([5, 5, 5])
        if nb_workers is None:
            nb_workers = 2
        return MasterGraph('master', initial_image=x, nb_workers=nb_workers)

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

            # assert g.subgraph('summation').tensor('target') is g.tensor('image')
