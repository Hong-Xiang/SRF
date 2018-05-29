import unittest
from srf.api.app import SRFApp


class TestSRFApp(unittest.TestCase):
    def test_parse_config(self):
        # config = {
        #     'scanner_type': None,
        #     # 'scanner': {'aaa': 5}
        # }
        scanner = DummyScanner(config)
        config = {
            'task_type': 'OSEM'
        }
        app = SRFApp(config, scanner)
        self.assertEqual(app.configure,
                         {'task_type': 'OSEM', })
