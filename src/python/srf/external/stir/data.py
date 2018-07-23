from dxl.data import DataClass
from srf.data import PETCylindricalScanner
from pathlib import Path


class STIRPETCylindericalScanner(PETCylindricalScanner):
    ...


class ReconstructionSpec(DataClass):
    __slots__ = ('path_sinogram_header',
                 'image_size_xy',
                 'nb_subsets',
                 'nb_subiterations')


class SinogramDataSpecScript:
    template = 'sinogram.hs.j2'

    def __init__(self, spec: STIRPETCylindericalScanner):
        self.spec = spec

    def render(self, template) -> str:
        return template.render(self.spec)


class ReconstructionSpecScript:
    template = 'OSMapOSL.par.j2'

    def __init__(self, spec: ReconstructionSpec):
        self.spec = spec

    def render(self, template) -> str:
        return template.render(spec=self.spec)
