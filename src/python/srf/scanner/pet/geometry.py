class RingGeometry():
    def __init__(self, config:dict):
        self.inner_radius = config['inner_radius']
        self.outer_radius = config['outer_radius']
        self.axial_radius = config['axial_length']
        self.nb_rings = config['nb_rings']
        self.nb_blocks_per_ring = config['nb_blocks_per_ring']
        self.gap = config['gap']
    