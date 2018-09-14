import unittest
import numpy as np
import tensorflow as tf
from srf.scanner import TestScanner

from srf.tensor import projection, image
from srf.task import TaskEfficiencyMap
from srf.model import TestBackProjection, BackProjection


class TestTaskEfficiencyMap(unittest.TestCase):

    def make_test_scanner(self):
        scanner = TestScanner()
        return scanner

        # scanner.get_scanner_projection(task)

    def test_init(self):
        scanner = self.make_test_scanner()
        task = TaskEfficiencyMap(scanner)
        self.assertIs(task.scanner, scanner)
        self.assertIsInstance(task.subgraph('backprojection'), BackProjection)
        # lors = Tensor(scanner.get_scanner_projection())
        self.assertIs(task.subgraph('backprojection').tensor['input'],
                      task.tensor('lors'))
        self.assertIs(task.tensor['effmap'],
                      task.subgraph('backprojection').tensor['output'])

    def test_make_lors(self):
        scanner = self.make_test_scanner()
        task = TaskEfficiencyMap(scanner)
        with tf.testing.test_session() as sess:
            result = sess.run(task.tensor['lors'])
        expect = scanner.get_projection()
        np.testing.assert_array_almost_equal(expect, result)

    def make_test_backprojector(self):
        return TestBackProjection()

    def test_compute_map(self):
        scanner = self.make_test_scanner()
        bp = self.make_test_backprojector()
        task = TaskEfficiencyMap(scanner, graphs={'backprojection': bp})
        result_map = task.compute_map()
        expected_map = bp.expected_output()
        np.testing.assert_array_almost_equal(
            result_map, expected_map, 'map values not equal!')

    def assertIsCalled(self, func):
        pass

    def test_run(self):
        scanner = self.make_test_scanner()
        task =  TaskEfficiencyMap(scanner)
        task.run()
        self.assertIsCalled(task.compute_map)
        self.assertIsCalled(task.save_map)
