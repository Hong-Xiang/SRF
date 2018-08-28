from jfs import Path
import numpy as np


def save_bin(target,data):
    data.tofile(target,'a')

def load_bin(target):
    data = np.fromfile(target,dtype='float32')
    return data
    