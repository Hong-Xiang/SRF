'''
This file provides methods to generate a GATE geometry macro file from the given srf unified 
The GATE system used to generate mac is ecat(). The 
'''
from srf.data import Ecat
from srf.external.castor.data import EcatScript
from srf.external.castor.io import save_script, render

__all__ = ['make_mac']


def get_scanner(config: dict):
    geo = config['ring']
    block = config['block']
    ecat = Ecat(geo['outer_radius'],
                geo['inner_radius'],
                geo['axial_length'],
                geo['nb_ring'],
                geo['nb_block_per_ring'],
                "LYSO",
                geo['gap'],
                block['size'],
                block['grid'])
    scanner_script = EcatScript(spec=ecat)
    return render(scanner_script)


def make_mac(config:dict):
    # print(config)
    scanner_str = get_scanner(config['scanner']['petscanner'])
    if 'name' in config['scanner'].keys():
        path = config['scanner']['name']
    else:
        path = "your_ecat"
    save_script(path, scanner_str, '.mac')
