import unittest
from srf.api.app import SRFApp
from pathlib import Path
import json
import numpy as np
from dxl.learn.utils.test_utils import sandbox


class TestSRFAppCustomTest(unittest.TestCase):
    @sandbox
    def test_recon_osem(self):
        from srf.utils.test_utils import path_test_resources
        from srf.scanner import ScannerFactory
        from srf.task import TaskFactory
        with open(Path(path_test_resources())/'recon_task.json', 'r') as fin:
            config = json.load(fin)

        # scanner_para  = config['scanner']
        # scanner  = ScannerFactory.make_scanner(scanner_para)

        task = TaskFactory.make_task(config)
        task.run()
        output =  task.config('output')["./mCT_osem_final.npy"]
        result = np.load(output)
        expected = np.load(Path(path_test_resources())/'mCT_osem_expected.npy')
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_effmap(self):
        from srf.utils.test_utils import path_test_resources
        from srf.scanner import ScannerFactory
        from srf.task import TaskFactory
        with open(Path(path_test_resources())/'map_task.json', 'r') as fin:
            config = json.load(fin)

        # scanner_para  = config['scanner']
        # scanner  = ScannerFactory.make_scanner(scanner_para)

        task = TaskFactory.make_task(config)
        task.run()
        output =  task.config('output')["./mCT_Map.npy"]
        result = np.load(output)
        expected = np.load(Path(path_test_resources())/'mCT_Map_expected.npy')
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_ring_map_single_ring_pair(self):
        pass

    def test_preprocessing(self):
        from srf.utils.test_utils import path_test_resources
        from srf.scanner import ScannerFactory
        from srf.task import TaskFactory
        with open(Path(path_test_resources())/'preprocess_task.json', 'r') as fin:
            config = json.load(fin)

        # scanner_para  = config['scanner']
        # scanner  = ScannerFactory.make_scanner(scanner_para)

        task = TaskFactory.make_task(config)
        task.run()
        output =  task.config('output')["./lors_processed.npy"]
        result = np.load(output)
        expected = np.load(Path(path_test_resources())/'lors_processed_expected.npy')
        np.testing.assert_array_almost_equal(result, expected)
        # config = {
        #     'scanner_type': None,
        #     # 'scanner': {'aaa': 5}
        # }
        # scanner = DummyScanner(config)
        # config = {
        #     'task_type': 'OSEM'
        # }
        # app = SRFApp(config, scanner)
        # self.assertEqual(app.configure,
        #                  {'task_type': 'OSEM', })




