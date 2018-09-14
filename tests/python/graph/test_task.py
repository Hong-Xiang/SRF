import unittest
# from srf.graph import Task

import pytest


@pytest.mark.skip('not impl yet')
class TestTask(unittest.TestCase):
    def test_basic(self):
        dummy_scanner = DummyScanner()
        dummy_master_worker = DummyMasterWorker()
        task = Task(dummy_scanner, dummy_master_worker)
        self.assertEqual(task.tensors['step1'], dummy_scanner['make_lors'])
        self.assertEqual(task.tensors['step2'], dummy_master_worker['init'])
        self.assertEqual(task.tensors['step3'], dummy_master_worker['recon'])

        class DummyScanner:
            def __getitem__(self, key):
                if key == 'make_lors':
                    return 12343254
