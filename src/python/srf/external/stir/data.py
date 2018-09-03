from doufo import dataclass, List
from srf.data import PETCylindricalScanner
from pathlib import Path


@dataclass
class SinogramSpec(PETCylindricalScanner):
    path_sinogram: str

    @property
    def ring_distance(self):
        return self.axial_length / self.nb_rings/self.nb_crystal_axial

    @property
    def nb_crystal_axial(self):
        return self.blocks[0].grid[2]

    @property
    def nb_crystal_transaxial(self):
        return self.blocks[0].grid[1]

    @property
    def axial_coordinate(self):
        rings = self.nb_rings*self.nb_crystal_axial
        coordinate='{'+str(rings)
        for i in range (1,rings):
            coordinate = coordinate+','+str(rings-i)+','+str(rings-i)
        coordinate = coordinate+'}'
        return coordinate

    @property
    def ring_difference(self):
        rings = self.nb_rings*self.nb_crystal_axial
        ring_difference = '{'+ str(0)
        for i in range(1,rings):
            ring_difference = ring_difference+','+str(-i)+','+str(i)
        ring_difference = ring_difference+'}'
        return ring_difference

@dataclass
class ReconstructionSpec:
    path_sinogram_header: str
    image_grid_xy: int
    nb_subsets: int
    nb_subiterations: int


class ReconstructionSpecScript:
    template = 'OSMapOSL.par.j2'

    def __init__(self, spec):
        self.spec = spec

    def render(self, template) -> str:
        return template.render(spec=self.spec)


class SinogramSpecScript:
    template = 'sinogram.hs.j2'

    def __init__(self, spec):
        self.spec = spec

    def render(self, template) -> str:
        return template.render(spec=self.spec)
