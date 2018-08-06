from doufo import dataclass, List
from srf.data import PETCylindricalScanner
from pathlib import Path


@dataclass
class SinogramSpec(PETCylindricalScanner):
    path_sinogram: str

    @property
    def ring_distance(self):
        return self.axial_length / self.nb_rings

    @property
    def nb_crystal_axial(self):
        return self.blocks[0].grid[1]

    @property
    def nb_crystal_transaxial(self):
        return self.blocks[0].grid[2]


@dataclass
class ReconstructionSpec:
    path_sinogram_header: str
    image_size_xy: float
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
