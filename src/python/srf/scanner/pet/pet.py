import numpy as np

import itertools
from typing import Iterable
from srf.scanner.pet.block import Block, BlockPair, RingBlock
from srf.scanner.pet.geometry import RingGeometry
from srf.scanner.pet.spec import TOF


class PETScanner():
    """ A PETScanner is an abstract class of PET detectors.

    PETScanner descripts the geometry of a PET scanner.

    Attributes:
        _name: a string names PETScanner.
        _modality: a string indicates the modality of the scanner (here is "PET").
        _block_proto: block discretization 
    """

    def __init__(self, name, block_proto: Block, tof: TOF):
        self._name = name
        self._modality = 'PET'
        self._block_proto = block_proto
        self._tof = tof

    @property
    def name(self):
        return self._name

    @property
    def modality(self):
        return self._modality

    @property
    def tof(self):
        return self._tof

    @property
    def block_proto(self):
        return self._block_proto

    def _make_lors(self):
        """
        """
        raise NotImplementedError


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

    def __init__(self, name, scanner_para: RingGeometry, block: Block, tof: TOF):

        super().__init__(name=name, block_proto=block, tof=tof)
        self._geometry = scanner_para
        self._rings = self._make_rings()

    @property
    def geometry(self):
        return self._geometry

    @property
    def inner_radius(self):
        return self.geometry.inner_radius

    @property
    def outer_radius(self):
        return self.geometry.outer_radius

    @property
    def nb_blocks_per_ring(self):
        return self.geometry.nb_blocks_per_ring

    @property
    def nb_rings(self):
        return self.geometry.nb_rings

    @property
    def gap(self):
        return self.geometry.gap

    @property
    def rings(self):
        return self._rings

    def _make_rings(self):
        """ Generate a list of rings in the cylindrical scanner.

        Create a ring list of the cylindrical PET scanner. 
        Each item of the list is a block list which cocntain the blocks
        contains the single ring.
        """
        nb_rings = self.geometry.nb_rings
        nb_blocks = self.nb_blocks_per_ring
        gap = self.gap
        ri = self.inner_radius
        ro = self.outer_radius
        block_size = self.block_proto.block_size
        grid = self.block_proto.grid
        # k_TWOPI_DEGREE = 360.0

        rings = []
        block_size_z = block_size[2]

        # z position bottom ring
        bottom_z = -(block_size_z + gap) * (nb_rings - 1) / 2

        # x offset of the block proto
        block_x_offset = (ri + ro) / 2

        # loop rings to create the block lists.
        for ir in range(nb_rings):
            block_z_offset = bottom_z + ir * (block_size_z + gap)
            # print('block_z_offset:', block_z_offset)
            pos = [block_x_offset, 0, block_z_offset]
            block_list: Iterable[RingBlock] = []

            # loop blocks in a ring
            for ib in range(nb_blocks):
                # phi = k_TWOPI_DEGREE / nb_blocks * ib
                # rad_z = phi / k_TWOPI_DEGREE * 2 * np.pi
                rad_z = ib*2*np.pi/nb_blocks
                block_list.append(RingBlock(block_size, grid, pos, rad_z))
            rings.append(block_list)
        return rings

    @classmethod
    def make_block_pairs(cls, ring1: list, ring2: list) -> list:
        """ Return the block pairs within two rings.

        Args:
            ring1: the first ring
            ring2: the second ring
        Returns:
            block_pairs: a list of block pair connect the two rings.

        """
        block_pairs = []
        if ring1 is ring2:
            # if same ring,
            block_pairs = [BlockPair(b1, b2) for i1, b1 in enumerate(ring1)
                           for i2, b2 in enumerate(ring2) if i1 < i2]
        else:
            block_pairs = [BlockPair(b1, b2) for i1, b1 in enumerate(ring1)
                           for i2, b2 in enumerate(ring2) if i1 != i2]
        return block_pairs

    @classmethod
    def make_ring_pairs_lors(cls, ring1: list, ring2: list) -> np.ndarray:
        """

        """
        block_pairs = cls.make_block_pairs(ring1, ring2)
        lors = [bp.make_lors() for bp in block_pairs]
        return np.array(lors).reshape(-1, 6)


class MultiPatchPET(PETScanner):
    """ A MultiPatchPET is a PET scanner which is constructed by multiple patches.

    MultiPatchPET descripts the irregular PET scanner constructed by irregular patches.

    Attributes:
        _patch_list: list of patches to construct the whole scanner

    """

    def __init__(self):
        pass
