from doufo import dataclass, Vector, List
import numpy as np


@dataclass
class Block(DataClass):
    size: Vector
    grid: Vector

    @property
    def crystal_width(self):
        return self.size[1] / self.grid[1]

# TODO PETCylindricalScanner: use blocks instead of block,
# change some argument of __init__ into property.


@dataclass
class PETCylindricalScanner(DataClass):
    inner_radius: float
    outer_radius: float
    axial_length: float
    nb_rings: int
    nb_blocks_per_ring: int
    gap: float
    blocks: List[Block]

    @property
    def central_bin_size(self):
        return 2 * np.pi * self.inner_radius / self.nb_detectors_per_ring / 2

    @property
    def nb_detectors_per_ring(self):
        return self.nb_detectors_per_block * self.nb_blocks_per_ring

    @property
    def nb_detectors_per_block(self):
        return self.blocks[0].grid[1]

    @property
    def block_width(self):
        return self.blocks[0].size[1]

    @property
    def crystal_width(self):
        return self.blocks[0].crystal_width
