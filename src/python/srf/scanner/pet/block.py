import numpy as np

from dxl.shape.rotation.matrix import axis_to_axis
from dxl.shape.utils.vector import Vector3
from dxl.shape.utils.axes import Axis3, AXIS3_X, AXIS3_Z

# from srf.scanner.geometry import Vec3


class Block(object):
    def get_meshes(self):
        pass


class RingBlock(Block):
    """
    A RingBlock the conventional ring geometry of scanner.
    A RingBlock is always towards the center of the z axis of the ring and the 
    inner face is parallel to Z axis. The geometry of RingBlock is decide by size, center, grid and the angle rotated 
    by Z axis. 

    Note: the center of the position before rotated.

    Attrs:
        _block_size: the size of the block.
        _center: position of the block center.
        _grid: discrete meshes of the block.
        _rad_z: the angle which indicates the 
    """

    def __init__(self, block_size , center, grid, rad_z: np.float32):
        self._block_size = np.array(block_size)
        self._center = np.array(center)
        self._grid = np.array(grid)
        self._rad_z = rad_z

    @property
    def grid(self):
        return self._grid

    @property
    def center(self):
        return self._center

    @property
    def rad_z(self):
        return self._rad_z

    @property
    def block_size(self):
        return self._block_size

    def get_meshes(self) -> np.array:
        """
        return all of the crystal centers in a block
        """

        interval = self.block_size / self.grid
        grid = self.grid

        p_start = self.center - self.block_size + interval/2
        p_end = self.center + self.block_size - interval/2

        mrange = [np.linspace(p_start[i], p_end[i], grid[i]) for i in range(3)]

        meshes = np.array(np.meshgrid(mrange[0], mrange[1], mrange[2]))

        # print(meshes.shape)
        meshes = np.transpose(meshes)
        source_axis = AXIS3_X
        target_axis = Axis3(
            Vector3([np.cos(self.rad_z), np.sin(self.rad_z), 0]))
        rot = axis_to_axis(source_axis, target_axis)

        rps = rot @ np.reshape(meshes, (3, -1))
        return np.transpose(rps)


# class PatchBlock(Block):
#     raise NotImplementedError
