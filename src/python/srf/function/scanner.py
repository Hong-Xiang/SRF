from ..scanner.pet import RingGeometry,Block,TOF,CylindricalPET

def make_scanner(scanner_class, config):
    if scanner_class == 'Cylinder':
        ring = RingGeometry(config['ring'])
        block = Block(block_size=config['block']['size'],
                  grid=config['block']['grid'])
        name = config['name']
        tof = TOF(res=config['tof']['resolution'], bin=config['tof']['bin'])
        return CylindricalPET(name, ring, block, tof)
    