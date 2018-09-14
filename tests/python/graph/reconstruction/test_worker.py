from srf.test import TestCase
import pytest
from dxl.learn.core import Model, Variable, Constant, SubgraphMakerFactory
from srf.graph.reconstruction import WorkerGraph, OSEMWorkerGraph
import numpy as np


class WorkerGraphTestCase(TestCase):
    def setUp(self):
        super().setUp()
        SubgraphMakerFactory.register(
            'worker/reconstruction', self.dummpy_recon_step_maker())

    def get_x_and_target(self):
        x = Variable('x', initializer=np.ones([5] * 3, dtype=np.float32))
        t = Variable('t', initializer=np.ones([5] * 3, dtype=np.float32))
        return x, t

    def get_subset(self):
        s = Constant(1, 'subset')
        return s

    def dummpy_recon_step_maker(self):
        class ReconStep(Model):
            def __init__(self, info, inputs):
                super().__init__(info, inputs=inputs)

            def kernel(self, inputs):
                return Constant(5 * np.ones([5] * 3, dtype=np.float32), 'result')

        def maker(inputs):
            return lambda g, n: ReconStep(g.info.child_scope(n), inputs={
                'image': inputs['x']
            })

        return maker

    def get_loader(self):
        class DummyLoader:
            def __init__(self, name=None):
                pass

            def load(self, target_graph):
                return {'projection_data': Constant(np.ones([30, 7]), 'projection_data')}, ()

            def to_split(self, target_graph):
                return ('projection_data',)

        return DummyLoader()

    def get_dummpy_recon_step_cls(self, x):
        class DummyReconStep(Model):
            def kernel(self, inputs):
                return x * 5
        return DummyReconStep


class TestWorkerGraph(WorkerGraphTestCase):
    def get_graph_and_inputs(self):
        x, t = self.get_x_and_target()

        return WorkerGraph('worker', x, t, loader=self.get_loader(), recon_step_cls=self.get_dummpy_recon_step_cls(x)), {'x': x, 'target': t}

    def test_recon_model_x_linked(self):
        g, inputs = self.get_graph_and_inputs()
        assert g.subgraph(g.KEYS.GRAPH.RECONSTRUCTION).tensor(
            'image').data is inputs['x'].data

    def test_recon_model_image_type(self):
        from srf.tensor import Image
        g, inputs = self.get_graph_and_inputs()
        assert isinstance(g.subgraph(g.KEYS.GRAPH.RECONSTRUCTION).tensor(
            'image'), Image)

    def test_x_target_linked(self):
        g, inputs = self.get_graph_and_inputs()
        assert g.tensor('target') is inputs['target']

    def test_recon_model_output_linked_with_target(self):
        g, inputs = self.get_graph_and_inputs()
        assert g.tensor('target') is g.tensor('update').target
        assert g.subgraph('reconstruction').tensor(
            'main') is g.tensor('update').source

    def test_run_updated_target(self):
        g, inputs = self.get_graph_and_inputs()
        with self.variables_initialized_test_session() as sess:
            sess.run(g.tensor(g.KEYS.TENSOR.UPDATE))
            self.assertFloatArrayEqual(
                sess.run(inputs['target']), 5 * np.ones([5] * 3))


class TestOSEMWorkerGraph(WorkerGraphTestCase):
    def get_graph_and_inputs(self):
        x, t = self.get_x_and_target()
        s = self.get_subset()
        return OSEMWorkerGraph('worker', x, t, s, loader=self.get_loader(), nb_subsets=3, recon_step_cls=self.get_dummpy_recon_step_cls(x)), {'x': x, 'target': t, 'subset': s}

    def test_subset_linked(self):
        g, inputs = self.get_graph_and_inputs()
        inputs = g._construct_inputs()
        assert tuple(inputs['projection_data'].shape) == (10, 7)

    def test_run_subset_value(self):
        g, inputs = self.get_graph_and_inputs()
        model_inputs = g._construct_inputs()
        dummy_lors = np.zeros([30, 7])
        for i in range(30):
            dummy_lors[i, :] = i
        with self.variables_initialized_test_session() as sess:
            for s in range(3):
                feeds = {
                    g.tensor('projection_data'): dummy_lors,
                    inputs['subset']: s,
                }
                expected = np.zeros([10, 7])
                for i in range(10):
                    expected[i, :] = (10 * s) + i
                self.assertFloatArrayEqual(
                    sess.run(model_inputs['projection_data'], feeds), expected)
