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

    x = Vector3(1,0,0)
    print(x.value)
    a = Rotation()
    a.set_rotation_matrix((math.pi)/4, Axis.x)
    print(a.rotation_matrix)
    y = a.rotate(x)
    print (y.value)


