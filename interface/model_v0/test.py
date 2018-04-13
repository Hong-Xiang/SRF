# from interface import Dimension
from tensor import Vector3
from interface import Rotation, Axis
import numpy as np
import math
# class Test:
#     def __init__(self, dim: Dimension):
#         self.dimension = dim    

#     def dim(self):
#         return self.dim


if __name__ is "__main__":
    position = Vector3(0, 0, 0)
    print('position: ')
    print(position)
    pixelsize = Vector3(1, 1, 1)
    grid = Vector3(100, 100, 100)
    orientation = (1, 0, 0)

    from interface import Discretization,PhysicsCartesian
    discretization = Discretization()
    img_phy_c = PhysicsCartesian(discretization,grid,orientation,pixelsize,position)
    from pet import EfficiencyMap
    data = np.zeros([100,100,100])
    map = EfficiencyMap(data,img_phy_c)
    print('construction completed!')