from interface import Dimension
from tensor import Info

class Test:
    def __init__(self, dim: Dimension):
        self.dimension = dim    

    def dim(self):
        return self.dim


if __name__ is "__main__":
    dim1 = Dimension(3)
    a = Test()
    
    print(a.test())


