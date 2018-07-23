from dxl.data import DataClass
from srf.data import PETCylindricalScanner
from pathlib import Path


class STIRPETCylindericalScanner(PETCylindricalScanner):
    ...


class STIRReconstructionSpec(DataClass):
    __slots__ = ('path_sinogram_header',
                 'image_size_xy',
                 'nb_subsets',
                 'nb_subiterations')




class SinogramDataSpecScript:
    template = Path(__file__) / '..' / 'template' / 'sinogram.hs.j2'

    def render(self, spec: STIRPETCylindericalScanner) -> str:
        pass


class ReconstructionSpecScript:
    template = Path(__file__) / '..' / 'template' / 'OSMApOSL.par.j2'

    def render(self, spec: STIRReconstructionSpec) -> str:
        pass
