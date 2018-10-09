import numpy as np
from numpy import matlib
from scipy.optimize import curve_fit

__all__ = ['gaussian1d', 'gaussian2d', 'gaussFit1d', 'gaussFit2d', 'gaussFit1d_solver', 'gaussFit2d_solver']

def gaussian1d(x, a, sigma):
    if isinstance(x, list):
        x = np.array(x)
    return a * np.exp(-x ** 2 / (2 * sigma ** 2))

def gaussian2d(data, a, sigma):
    x = data[0]
    y = data[1]
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(y, list):
        y = np.array(y)
    r = np.sqrt(x ** 2 + y ** 2)
    return gaussian1d(r, a, sigma)

def gaussFit1d(x, yn):
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(yn, list):
        yn = np.array(yn)
    popt, _ = curve_fit(gaussian1d, x, yn)

    return float(popt[0]), float(popt[1])
#     print(popt)

def gaussFit2d(data, zn):
    x = data[0]
    y = data[1]
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(y, list):
        y = np.array(y)
    if isinstance(zn, list):
        zn = np.array(zn)
    r = np.sqrt(x ** 2 + y ** 2)
    
    return gaussFit1d(r, zn)

def gaussFit1d_solver(x, yn):
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(yn, list):
        yn = np.array(yn)
    def func(r2, alpha, beta):
        return alpha + beta * r2
    r2, out = [], []
    for i in range(len(x)):
        if yn[i] > 1e-7:
            r2 = r2 + [x[i] ** 2]
            out = out + [np.log(yn[i])]
    r2 = np.array(r2)
    out = np.array(out)
    popt, _ = curve_fit(func, r2, out)
    return float(np.exp(popt[0])), float(np.sqrt(-0.5 / popt[1]))



def gaussFit2d_solver(xy, zn):
    x = xy[0]
    y = xy[1]
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(y, list):
        y = np.array(y)
    if isinstance(zn, list):
        zn = np.array(zn)
    def func(r2, alpha, beta):
        return alpha + beta * r2
    r2, out = [], []
    for i in range(len(x)):
        if zn[i] > 1e-7:
            r2 = r2 + [x[i] ** 2 + y[i] ** 2]
            out = out + [np.log(zn[i])]
    r2 = np.array(r2)
    out = np.array(out)
    popt, _ = curve_fit(func, r2, out)
    return float(np.exp(popt[0])), float(np.sqrt(-0.5 / popt[1]))