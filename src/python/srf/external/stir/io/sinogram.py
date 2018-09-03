import numpy as np
import matplotlib.pyplot as plt
import pylab

def save_sinogram(path, data):
    inner_data = data.unbox()   
    output = inner_data.astype(np.float32)
    output.tofile(path)