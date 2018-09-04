from doufo import dataclass, List
from srf.data import PETCylindricalScanner
from pathlib import Path


@dataclass
class TOFSpec:
    tof_flag: int
    tof_resolution: float
    tof_binsize: float
    tof_limit: int

@dataclass
class IterSpec:
    nb_subiterations: int
    start_iteration: int


@dataclass
class ScannerSpec(PETCylindricalScanner):
    @property
    def block_size(self):
        return str(self.blocks[0].size[0])+' '+str(self.blocks[0].size[1])+' '+str(self.blocks[0].size[2])

    @property
    def block_grid(self):
        return str(self.blocks[0].grid[0])+' '+str(self.blocks[0].grid[1])+' '+str(self.blocks[0].grid[2])

@dataclass
class ImageSpec:
    image_grid: list
    image_size: list
    @property
    def imagegrid(self):
        return str(self.image_grid[0])+' '+str(self.image_grid[1])+' '+str(self.image_grid[2])

    @property
    def imagesize(self):
        return str(self.image_size[0])+' '+str(self.image_size[1])+' '+str(self.image_size[2])


@dataclass
class ReconstructionSpec(ImageSpec):
    path_input: str
    path_output: str    
    path_map: str    
    start_iteration: int
    nb_subiterations: int
    tof_flag: int
    tof_resolution: float
    tof_binsize: float
    tof_limit: int
    abf_flag: int

    @property
    def start_name(self):
        if self.start_iteration ==1:
            name = 'None'
        else:
            name = self.path_output+str(self.start_iteration-1)
        return name
        

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
