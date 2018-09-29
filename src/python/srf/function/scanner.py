from ..scanner.pet import RingGeometry,Block,TOF,CylindricalPET
from ..data import ScannerClass

def make_scanner(scanner_class:ScannerClass, config):
    if scanner_class == ScannerClass.CylinderPET:
        config = modified_ring_scanner(config)
        ring = RingGeometry(config['ring'])
        block = Block(block_size=config['block']['size'],
                  grid=config['block']['grid'])
        #name = config['name']
        name = 'mCT'
        if 'tof' in config:
            tof = TOF(res=config['tof']['resolution'], bin=config['tof']['bin'])
        else:
            tof = None
        return CylindricalPET(name, ring, block, tof)

def modified_ring_scanner(config):
    config['ring']['nb_rings'] = config['ring']['nb_rings']*config['block']['grid'][2]
    config['block']['size'][2] = config['block']['size'][2]/config['block']['grid'][2]
    config['block']['grid'][2] = 1
    return config
    