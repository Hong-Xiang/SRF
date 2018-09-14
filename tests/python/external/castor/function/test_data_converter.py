import pytest
from srf.test import TestCase
from pathlib import Path
from srf.external.castor.function.listmode2cdhf import generate_cdh

class ConverterTestBase(TestCase):
    def setUp(self):
        super().setUp()


class TestDataConverter(ConverterTestBase):

    def get_config_path(self):
        config_path = ("/home/chengaoyu/UnitTestResource/SRF/external/castor/utility")
        return config_path

    @pytest.mark.skip(reason='TODO str operation')
    def test_generate_cdh(self):
        # config = self.get_config_path()
        expected_file_name = "mydata"
        expected_data_str = "fasf\n"
        result_file_name, result_data_str = generate_cdh()
        assert expected_file_name == result_file_name
        assert expected_data_str.split('\n') == result_data_str.split('\n')

        
