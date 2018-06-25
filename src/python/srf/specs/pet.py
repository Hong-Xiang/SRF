from srf.specs.data import Specs


class RingSpec(Specs):
    FIELDS = ('inner_radius', 'outer_radius',
              'axial_length', 'nb_ring',
              'nb_block_ring', 'gap')

class PatchSpec(Specs):
    FIELDS = ('path_file')


class PETScannerSpec(Specs):
    FIELDS = ('modality', 'name', 'block')


class CylindricalPETSpec(PETScannerSpec):
    class KEYS:
        RING = 'ring'
    FIELDS = tuple(list(PETScannerSpec.FIELDS) + [KEYS.RING,])

class MultiPatchPETSpec(PETScannerSpec):
    class KEYS:
        PATCHFILE = 'patch_file'
    FIELDS = tuple(list(PETScannerSpec.FIELDS),[KEYS.PATCHFILE])

