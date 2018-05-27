from ..specs import pet
from dxl.core.config import ConfigurableWithName

import numpy as np 


class PETScanner(ConfigurableWithName):
    """ A PETScanner is an abstract class of PET detectors.
    
    PETScanner descripts the geometry of a PET scanner.

    Attributes:
        name: a string names PETScanner.
        modality: a string indicates the modality of the scanner (here is "PET").
    """

    def __init__(self, ):
        pass
    
    def _make_lors(self):
        """
        """
        pass
    
class CylindricalPET(PETScanner):
    """ A CylindricalPET is a conventional cylindrical PET scanner.

    Attributes:
        _inner_radius: the inner radius of scanner.
        _outer_radius: the outer radius of scanner.
        _axial_length: the axial length of scanner.
        _nb_ring: number of rings.
        _nb_block_ring: number of blocks in a single ring.
        _gap:  the gap bewtween rings.
        _rings: list of block list ring by ring. 
    """

    def __init__(self, specs:pet.CylindricalPETSpec):
        
        pass
    
    def _make_rings(self):
        """ 
        """
        pass
    
    def _make_block_pairs(self, ring1:int, ring2: int) -> list: 
        """ 
        """
        pass
    
    def make_ring_pairs_lors(self, ring1:int, ring2:int) -> np.ndarray: 
        pass

class MultiPatchPET(PETScanner):
    """ A MultiPatchPET is a PET scanner which is constructed by multiple patches.
    
    MultiPatchPET descripts the irregular PET scanner constructed by irregular patches.

    Attributes:
        _patch_list: list of patches to construct the whole scanner


    """
    pass