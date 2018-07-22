from dxl.data import DataClass
import numpy as np


class Block(DataClass):
    __slots__ = ('size', 'grid')

    @property
    def crystal_width(self):
        return self.size[1] / self.grid[1]


class PETCylindricalScanner(DataClass):
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
    def nb_detectors_per_ring(self):
        return self.nb_detectors_per_block * self.nb_blocks_per_ring

    @property
    def nb_detectors_per_block(self):
        return self.block.grid[1]

    @property
    def block_width(self):
        return self.block.size[1]

    @property
    def crystal_width(self):
        return self.block.crystal_width
