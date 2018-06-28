import numpy as np
class TOF():
    def __init__(self, res:np.float32, bin: np.float32):
        self.resolution = res
        self.bin = bin
