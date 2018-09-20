from ..scanner.pet import RingGeometry,Block,TOF,CylindricalPET
from ..data import ScannerClass

def make_scanner(scanner_class:ScannerClass, config):
    if scanner_class == ScannerClass.CylinderPET:
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
    