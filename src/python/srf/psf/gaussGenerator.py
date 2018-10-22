import numpy as np
from numpy import matlib

__all__ = ['gauss2DGenerator', 'gauss1DGenerator', 'imageGaussGenerator']

def gauss2DGenerator(sigma = 1, size = [8,8], dot = []):

    if not isinstance(size, list) and not isinstance(size, int):
        raise TypeError('Gaussion Kernel should have integer size or list[int] shape')
    if isinstance(size, int):
        size = [size, size]
    if not dot:
        center_x, center_y = size[0] / 2.0 - 0.5, size[1] / 2.0 - 0.5
    else:
        center_x, center_y = dot[0] + 0.5, dot[1] + 0.5
    mask_x = np.matlib.repmat(center_x, size[0], size[1])
    mask_y = np.matlib.repmat(center_y, size[0], size[1])

    x1 = np.arange(size[0])
    x_map = np.matlib.repmat(x1, size[1], 1)
    y1 = np.arange(size[1])
    y_map = np.matlib.repmat(y1, size[0], 1)
    y_map = np.transpose(y_map)

    Gauss_map = (x_map-mask_x)**2+(y_map-mask_y)**2
    Gauss_map = np.exp(-0.5*Gauss_map/sigma/sigma)
    return Gauss_map

# def gauss2D(sigma = 1, size = 8, dot = []):
    
#     if not isinstance(size, int):
#         raise TypeError('Gaussion Kernel should have integer size')
#     if not dot:
#         center_x, center_y = size / 2.0 - 0.5, size / 2.0 - 0.5
#     else:
#         center_x, center_y = dot + 0.5, dot + 0.5
#     mask_x = np.matlib.repmat(center_x, size, size)
#     mask_y = np.matlib.repmat(center_y, size, size)
    
#     x1 = np.arange(size)
#     x_map = np.matlib.repmat(x1, size, 1)
#     y1 = np.arange(size)
#     y_map = np.matlib.repmat(y1, size, 1)
#     y_map = np.transpose(y_map)
    
#     Gauss_map = np.sqrt((x_map-mask_x)**2+(y_map-mask_y)**2)
#     Gauss_map = np.exp(-0.5*Gauss_map/sigma/sigma)
#     return Gauss_map

def gauss1DGenerator(sigma = 1, size = 8, dot = None):
    if not isinstance(size, int):
        raise TypeError('Gaussion Kernel should have integer size')
    if not dot:
        center = size / 2.0 - 0.5
    else:
        center = dot + 0.5
    mask_x = np.matlib.repmat(center, 1, size)

    x_map = np.arange(size)

    Gauss_map = (x_map-mask_x)**2
    Gauss_map = np.exp(-0.5*Gauss_map/sigma/sigma)
    return Gauss_map

def imageGaussGenerator(dots = [], sigma = [], size = [8, 8]):
    img = np.array([[0 for _ in range(size[0])] for _ in range(size[1])])
    if not dots:
        return img
    for i, dot in zip(range(len(dots)), dots):
        if not sigma:
            tmp_img = gauss2DGenerator(size = size, dot = dot)
        else:
            tmp_img = gauss2DGenerator(sigma = sigma[i], size = size, dot = dot)
        img = img + tmp_img
    return img