from srf.test import TestCase
from dxl.learn.test import OpRunSpy
import pytest


@pytest.mark.skip('not impl yet')
class TestMasterWorkerReconstructionGraph(TestCase):
    def make_graph(self, nb_workers):
        c = make_cluster(nb_workers=nb_workers)
        return MasterWorkerReconstructionGraph(cluster=c)

    def test_graphs_types(self):
        mg = self.make_graph()
        self.assertIsInstance(mg.subgraph('master'), MasterGraph)
        for i in range(mg.nb_workers):
            self.assertIsInstance(mg.subgraph('worker')[
                                  i], WorkerGraph)

    def test_image_linked(self):
        mg = self.make_graph()
        for i in range(mg.nb_workers):
            assert mg.subgraph('master').tensor(
                'image') is mg.subgraph('worker')[i].tensor('image')

    def test_buffer_linked(self):
        mg = self.make_graph()
        for i in range(mg.nb_workers):
            assert mg.subgraph('master').tensor(
                'buffer')[i] is mg.subgraph('worker')[i].tensor('buffer')

    def test_barrier_init_master(self):
        g = self.make_graph('master', 0)
        assert g.subgraph('init_barrier').tensor(
            'task') is g.subgraph('master').tensor('init')

    def test_barrier_init_worker(self):
        g = self.make_graph('master', 1)
        assert g.subgraph('init_barrier').tensor(
            'task') is g.subgraph('worker').tensor('init')

    def test_barrier_recon(self):
        pass

    def test_barrier_merge(self):
        pass

    def test_run(self):
        mg = self.make_graph()
        init_spy = OpRunSpy()
        recon_spy = OpRunSpy()
        merge_spy = OpRunSpy()
        g.tensors['init'] = init_spy.op
        g.tensors['recon'] = recon_spy.op
        g.tensors['merge'] = merge_spy.op
        with self.variables_initialized_test_session() as sess:
            assert sess.run(init_spy.nb_called) == 0
            assert sess.run(recon_spy.nb_called) == 0
            assert sess.run(merge_spy.nb_called) == 0
            g.run()
            assert sess.run(init_spy.nb_called) == 1
            assert sess.run(recon_spy.nb_called) == g.nb_iteraions
            assert sess.run(merge_spy.nb_called) == g.nb_iteraions
