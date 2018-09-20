from doufo import dataclass,  List
from doufo.tensor import Vector
import numpy as np
from enum import Enum

@dataclass
class Block:
    size: Vector
    grid: Vector


    @property
    def crystal_thickness(self):
        return self.size[0]/self.grid[0]

    @property
    def crystal_width(self):
        return self.size[1] / self.grid[1]

    @property
    def crystal_height(self):
        return self.size[2]/self.grid[2]

# TODO PETCylindricalScanner: use blocks instead of block,
# change some argument of __init__ into property.


@dataclass
class PETCylindricalScanner:
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

@dataclass
class Ecat:
    outer_radius: float
    inner_radius: float
    axial_length: float
    nb_ring: int
    nb_block_per_ring: int 
    material:str 
    gap: float
    block_size:Vector
    block_grid:Vector

class ScannerClass(Enum):
    CylinderPET = 0
    PolyhedronPET = 1
    ConebeamCT = 2
