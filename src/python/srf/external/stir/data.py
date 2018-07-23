from dxl.data import DataClass
from srf.data import PETCylindricalScanner
from pathlib import Path


class SinogramSpec(PETCylindricalScanner):
    __slots__ = tuple(list(PETCylindricalScanner.__slots__)
                      + ['path_sinogram'])

    @property
    def ring_distance(self):
        return self.axial_length / self.nb_rings

    @property
    def nb_crystal_axial(self):
        return self.block.grid[1]

    @property
    def nb_crystal_transaxial(self):
        return self.block.grid[2]


class ReconstructionSpec(DataClass):
    __slots__ = ('path_sinogram_header',
                 'image_size_xy',
                 'nb_subsets',
                 'nb_subiterations')


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
