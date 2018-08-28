from doufo import dataclass, List
from srf.data import PETCylindricalScanner
from pathlib import Path


@dataclass
class ScannerSpec(PETCylindricalScanner):
    @property
    def block_size(self):
        return self.blocks.size

    @property
    def block_grid(self):
        return self.blocks.grid

@dataclass
class ImageSpec:
    image_grid: list
    image_size: list

@dataclass
class ReconstructionSpec(ImageSpec):
    path_input: str
    path_output: str    
    path_map: str
    start_iteration: int
    nb_iterations: int
    tof_flag: int
    tof_resolution: float
    tof_binsize: float
    tof_limit: int
    abf_flag: int

@dataclass
class MapSpec(ImageSpec):
    path_map: str


class ReconstructionSpecScript:
    template = 'recon_task.txt.j2'

    def __init__(self, spec):
        self.spec = spec

    def render(self, template) -> str:
        return template.render(spec=self.spec)

class MapSpecScript:
    template = 'map_task.txt.j2'
    def __init__(self, spec):
        self.spec = spec

    def render(self, template) -> str:
        return template.render(spec=self.spec)


class ScannerSpecScript:
    template = 'ringpet_config.txt.j2'

    def __init__(self, spec):
        self.spec = spec

    def render(self, template) -> str:
        return template.render(spec=self.spec)
