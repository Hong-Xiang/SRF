import numpy as np
import matplotlib.pyplot as plt
import json
from dxl.data.io import load_npz
from srf.model._backprojection import BackProjectionToR
from srf.tensor import Image
from srf.physics import ToRModel
from dxl.learn.core import Constant
from dxl.learn.core import Session

pm = ToRModel('projection_model', kernel_width=3.86,
              tof_bin=6.0, tof_sigma2=162.303)

projection_data = {a: Constant(lors[a], a) for a in pm.AXIS}

projection_value = {a: Constant(
    np.ones([lors[a].shape[0]], dtype=np.float32), a) for a in pm.AXIS}

image = Image(Constant(np.ones([195, 195, 416], dtype=np.float32), 'image'), center=[
              0.0] * 3, size=[666.9, 666.9, 1422.72])

bpm = BackProjectionToR(
    'bpm', {'lors': projection_data, 'lors_value': projection_value}, image)

sess = Session()

bpi = sess.run(bpm())

print(bpi.shape, np.min(bpi), np.max(bpi))

plt.imshow(bpi[:, :, 216])
