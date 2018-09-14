from srf.specs.data import Specs

class ToFSpec(Specs):
    FIELDS = ('tof_res', 'tof_bin')

class BlockSpec(Specs):
    FIELDS = ('block_size', 'grid', 'interval')




class RingSpec(Specs):
    FIELDS = ('inner_radius', 'outer_radius',
              'axial_length', 'nb_ring',
              'nb_block_ring', 'gap')


class PatchSpec(Specs):
    FIELDS = ('path_file')


class PETScannerSpec(Specs):
    
    class KEYS:
        BLOCK = 'block'
        TOF = 'tof'
    FIELDS = ('modality', 'name', KEYS.BLOCK, KEYS.TOF)

    def __init__(self, config):
        super().__init__(config)
    
    def parse(self, key, cls):
        if key in self.data:
            self.data[key] = cls(self.data[key])





class CylindricalPETSpec(PETScannerSpec):
    class KEYS:
        RING = 'ring'
    FIELDS = tuple(list(PETScannerSpec.FIELDS) + [KEYS.RING,])

    def __init__(self, config):
        super().__init__(config)


class MultiPatchPETSpec(PETScannerSpec):
    class KEYS:
        PATCHFILE = 'patch_file'
    FIELDS = tuple(list(PETScannerSpec.FIELDS)+[KEYS.PATCHFILE,])

