from dxl.data import DataClass
import numpy as np


class Block:
    __slots__ = ('block_size',
                 'grid')


class CylindricalScanner:
    __slots__ = ('inner_radius',
                 'outer_radius',
                 'axial_length',
                 'nb_rings',
                 'nb_blocks_per_ring',
                 'gap',
                 'block')

    @property
    def central_bin_size(self):
        return 2 * np.pi * self.inner_radius / self.nb_crystals_per_ring / 2

    @property
    def nb_crystals_per_ring(self):
        pass
