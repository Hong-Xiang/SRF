import numpy as np
import itertools

from dxl.shape.function.rotation import axis_to_axis

from dxl.shape.data import Axis as Axis3
from dxl.shape.data.axis import AXIS3_X, AXIS3_Z
from doufo.tensor import transpose, Tensor


class Block(object):
    """
    A Block is a base class to present the geometry of PET scanner blocks.
    A block is a 3D discrete Cartesian meshgrid with size.
    The primary method of Block is to return the mesh center position of itself.
    Attrs:
        _block_size: the size of the block
        __grid: meshgrid of the block.

    """

    def __init__(self, block_size, grid):
        self._block_size = np.array(block_size)
        self._grid = np.array(grid)

    @property
    def block_size(self):
        return self._block_size

    @property
    def grid(self):
        return self._grid

    def get_meshes(self):
        raise NotImplementedError


class RingBlock(Block):
    """                        __ __ __                       
            z                 /__/__/__/|
            |   y            /__/__/__/||
            |  /             |__|__|__|||
            | /              |__|__|__|||
            |/__ __ __ x     |__|__|__|/


    A RingBlock the conventional ring geometry of scanner.
    A RingBlock is always towards the center of the z axis of the ring and the 
    inner face is parallel to Z axis. The geometry of RingBlock is decide by size, center, grid and the angle rotated 
    by Z axis. 

    Note: the center of the position before rotated.

    Attrs:
        _center: position of the block center.
        _rad_z: the angle which indicates the block orientation surrounding the Z axis.
    """

    def __init__(self, block_size, grid, center, rad_z: np.float32):
        super().__init__(block_size, grid)

        self._center = np.array(center)
        self._rad_z = rad_z

    @property
    def center(self):
        return self._center

    @property
    def rad_z(self):
        return self._rad_z

    def get_meshes(self) -> np.array:
        """ compute the mesh points of this block.
        Args: 
            None
        returns:
            rps: crystal centers positions in a block, and the positions 
                 is organized in an ndarray with the shape of [N, 3].  
        """

        interval = self.block_size / self.grid
        grid = self.grid

        p_start = self.center - (self.block_size - interval) / 2
        p_end = self.center + (self.block_size - interval) / 2

        [mrange_x, mrange_y, mrange_z] = [np.linspace(
            p_start[i], p_end[i], grid[i]) for i in range(3)]

        meshes = np.array(np.meshgrid(mrange_x, mrange_y, mrange_z))

        # print(meshes)
        # meshes = np.transpose(meshes)
        # source_axis = AXIS3_X
        # target_axis = Axis3(
        # Vector3([np.cos(self.rad_z), np.sin(self.rad_z), 0]))
        # target_axis = np.array([np.cos(self.rad_z), np.sin(self.rad_z), 0])
        # rot = axis_to_axis(source_axis, target_axis)
        # HACK for compat with dxshape
        rot = axis_to_axis([1.0, 0.0, 0.0], [np.cos(self.rad_z), np.sin(self.rad_z), 0])
        # rot = np.ones([3, 3])

        # print('type of rot!!!!!:', type(rot))
        # print(meshes.reshape((-1,3)))
        rps = rot.unbox()@np.reshape(meshes,(3, -1))
        # rps = Tensor(rot @ np.reshape(meshes, (3, -1)))
        # print(np.transpose(rps))
        return np.transpose(rps)


class PatchBlock(Block):
    pass


class BlockPair(object):
    """
    A BlockPair represents the a block pair in the PET scanner.

    Attrs:
        _block1: the 1st block
        _block2: the 2nd block

    Methods:
        get_lors(): connect the valid meshes of two blocks to create lors.
    """

    def __init__(self, block1, block2):
        self._block1 = block1
        self._block2 = block2

    @property
    def block1(self):
        return self._block1

    @property
    def block2(self):
        return self._block2

    def make_lors(self) -> np.ndarray:
        """ Compute all the lors of a block pair list. 
        Returns:
            lors: an np.ndarray with the shape of N*6, each column of which
                  contain the 3D position of two end points of an lor.
        """
        lors = []
        m0 = self.block1.get_meshes()
        m1 = self.block2.get_meshes()
        # HACK for 
        # m0, m1 = m0.unbox(), m1.unbox()

        lors = [x for x in itertools.product(m0, m1)]
        return lors
        # return np.array(lors).reshape(-1, 6)
