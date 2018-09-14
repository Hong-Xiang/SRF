import pytest
from srf.test import TestCase

from pathlib import Path
import json
from srf.external.castor.cli import parse_recon, parse_root_to_castor, parse_mac_to_geom


class ParserTestBase(TestCase):
    def setUp(self):
        super().setUp()


class TestCommandParser(ParserTestBase):

    def get_config_path(self):
        template_path = ("/home/chengaoyu/UnitTestResource/SRF/external/castor/utility")
        return template_path

    def test_parse_recon(self):
        expected_command_str = ("castor-recon -df data.cdh -opti MLEM " +
                                "-it 10:10 -proj classicSiddon " +
                                "-conv gaussian,3.0,3.0,3.0::psf -dim 100,100,100 " +
                                "-vox 3.0,3.0,3.0 -dout my_image")
        recon_config_file = self.get_config_path() + '/recon_config.json'
        # with open(recon_config_file, 'r') as fin:
        #     recon_config = json.load(fin)
        result_commmand_str = parse_recon(recon_config_file)
        assert result_commmand_str == expected_command_str

    def test_parse_root_to_castor(self):
        expected_command_str = ("castor-GATERootToCastor -i data.root " +
                                "-o data_name -m geo.mac -s scanner_name")
        converter_config_file = self.get_config_path() + '/root2castor_config.json'
        result_command_str = parse_root_to_castor(converter_config_file)
        assert result_command_str == expected_command_str

    def test_parse_mac_to_castor(self):
        expected_command_str = (
            "castor-GATEMacToGeom -m camera.mac -o scanner_name")
        geom_config_file = self.get_config_path() + '/mac2geom_config.json'
        result_command_str = parse_mac_to_geom(geom_config_file)
        assert result_command_str == expected_command_str

    @pytest.mark.skip(reason="NIY")
    def test_parse_lm_to_castor(self):
        pass
