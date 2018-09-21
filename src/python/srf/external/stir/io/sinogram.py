import numpy as np
import matplotlib.pyplot as plt
import pylab

def save_sinogram(path, data):
    inner_data = data.unbox() 
    output = np.array(inner_data,dtype='float32')
    output.tofile(path)