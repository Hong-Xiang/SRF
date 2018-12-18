# encoding: utf-8
'''
@author: Minghao Guo, Xiang Hong, Gaoyu Chen and Weijie Tao
@license: LGPL_v3.0
@contact: mh.guo0111@gmail.com
@software: srf_v2
@file: gaussian_fit.py
@date: 12/17/2018
@desc: new version of Scalable Reconstruction Framework for Medical Imaging
'''
import numpy as np
import scipy.optimize as opt

__all__ = ['fitgaussian_1d', 'fitgaussian_2d', 'fitgaussian_3d',
           'fitgaussian_1d_fix_mu', 'fitgaussian_2d_fix_mu', 'fitgaussian_3d_fix_mu',
           'fit_gaussian']
_threshold = 1e-16


def _gaussian_1d(amp, sig2, mu):
    return lambda x: amp * np.exp(-(x - mu) ** 2 / 2 / sig2)


def _gaussian_2d(amp, sigx2, sigy2, mux, muy):
    return lambda x, y: amp * np.exp(-(x - mux) ** 2 / 2 / sigx2) * \
                        np.exp(-(y - muy) ** 2 / 2 / sigy2)


def _gaussian_3d(amp, sigx2, sigy2, sigz2, mux, muy, muz):
    return lambda x, y, z: amp * np.exp(-(x - mux) ** 2 / 2 / sigx2) * \
                           np.exp(-(y - muy) ** 2 / 2 / sigy2) * \
                           np.exp(-(z - muz) ** 2 / 2 / sigz2)


def _gaussian_1d_fix_mu(amp, sig2):
    return lambda x: amp * np.exp(-x ** 2 / 2 / sig2)


def _gaussian_2d_fix_mu(amp, sigx2, sigy2):
    return lambda x, y: amp * np.exp(-x ** 2 / 2 / sigx2) * \
                        np.exp(-y ** 2 / 2 / sigy2)


def _gaussian_3d_fix_mu(amp, sigx2, sigy2, sigz2):
    return lambda x, y, z: amp * np.exp(-x ** 2 / 2 / sigx2) * \
                           np.exp(-y ** 2 / 2 / sigy2) * \
                           np.exp(-z ** 2 / 2 / sigz2)


def fit_gaussian(data, pos, mode = None, **kwargs):
    ndim = len(pos)
    if ndim > 3:
        ndim = 1
    if ndim == 1:
        if 'mu' in kwargs.keys():
            kmu = kwargs['mu']
            mu = kmu[0] if isinstance(kmu, tuple) else kmu
        else:
            mu = 0
        if isinstance(pos, tuple):
            pos = pos[0]

        x, data = pos.ravel(), data.ravel()
        x = x[data > _threshold]
        data = data[data > _threshold]
        if mode == 'fix_mu':
            def _error_function(p):
                return np.ravel(_gaussian_1d_fix_mu(*p)(x - mu) - data)

            if 'initial_guess' in kwargs.keys():
                init = kwargs['initial_guess']
            else:
                init = [np.max(data), 1]
            p = opt.leastsq(_error_function, init)
            return np.append(p[0], [mu])
        elif mode == 'fit_mu':
            def _error_function(p):
                return np.ravel(_gaussian_1d(*p)(x) - data)

            if 'initial_guess' in kwargs.keys():
                init = kwargs['initial_guess']
            else:
                init = [np.max(data), 1, mu]
            p = opt.leastsq(_error_function, init)
            return p[0]

        else:
            raise NotImplementedError
    elif ndim == 2:
        if 'mu' in kwargs.keys():
            kmu = kwargs['mu']
            mux, muy = kmu[0], kmu[1]
        else:
            mux = muy = 0

        x, y, data = pos[0].ravel(), pos[1].ravel(), data.ravel()
        maxv = np.max(data)
        x = x[data > _threshold * maxv]
        y = y[data > _threshold * maxv]
        data = data[data > _threshold * maxv]

        if mode == 'fix_mu':
            def _error_function(p):
                return np.ravel(_gaussian_2d_fix_mu(*p)(x - mux, y - muy) - data)

            if 'initial_guess' in kwargs.keys():
                init = kwargs['initial_guess']
            else:
                init = [np.max(data), 1, 1]
            p = opt.leastsq(_error_function, init)

            return np.append(p[0], [mux, muy])
        elif mode == 'fit_mu':
            def _error_function(p):
                return np.ravel(_gaussian_2d(*p)(x, y) - data)

            if 'initial_guess' in kwargs.keys():
                init = kwargs['initial_guess']
            else:
                init = [1e6, 1, 1, mux, muy]
            p = opt.leastsq(_error_function, init)
            return p[0]
        else:
            raise NotImplementedError
    elif ndim == 3:
        if 'mu' in kwargs.keys():
            kmu = kwargs['mu']
            mux, muy, muz = kmu[0], kmu[1], kmu[2]
        else:
            mux = muy = muz = 0

        x, y, z, data = pos[0].ravel(), pos[1].ravel(), pos[2].ravel(), data.ravel()
        maxv = np.max(data)
        x = x[data > _threshold * maxv]
        y = y[data > _threshold * maxv]
        z = z[data > _threshold * maxv]
        data = data[data > _threshold * maxv]
        if mode == 'fix_mu':
            def _error_function(p):
                return np.ravel(_gaussian_3d_fix_mu(*p)(x - mux, y - muy, z - muz) - data)

            if 'initial_guess' in kwargs.keys():
                init = kwargs['initial_guess']
            else:
                init = [np.max(data), 1, 1, 1]
            p = opt.leastsq(_error_function, init)
            return np.append(p[0], [mux, muy, muz])
        elif mode == 'fit_mu':
            def _error_function(p):
                return np.ravel(_gaussian_3d(*p)(x, y, z) - data)

            if 'initial_guess' in kwargs.keys():
                init = kwargs['initial_guess']
            else:
                init = [np.max(data), 1, 1, 1, mux, muy, muz]
            p = opt.leastsq(_error_function, init)
            return p[0]

        else:
            raise NotImplementedError
    else:
        raise NotImplementedError


def fitgaussian_1d(data, x, pos):
    x, data = x.ravel(), data.ravel()
    x = x[data > _threshold]
    data = data[data > _threshold]

    def _error_function(p):
        return np.ravel(_gaussian_1d(*p)(x) - data)

    return opt.leastsq(_error_function, np.array([1, pos, 1]))


def fitgaussian_2d(data, x, y, pos):
    x, y, data = x.ravel(), y.ravel(), data.ravel()
    x = x[data > _threshold]
    y = y[data > _threshold]
    data = data[data > _threshold]

    def _error_function(p):
        return np.ravel(_gaussian_2d(*p)(x, y) - data)

    return opt.leastsq(_error_function, np.array([1, pos[0], pos[1], 1, 1]))


def fitgaussian_3d(data, x, y, z, pos):
    x, y, z, data = x.ravel(), y.ravel(), z.ravel(), data.ravel()
    x = x[data > _threshold]
    y = y[data > _threshold]
    z = z[data > _threshold]
    data = data[data > _threshold]

    def _error_function(p):
        return np.ravel(_gaussian_3d(*p)(x, y, z) - data)

    return opt.leastsq(_error_function, np.array([1, pos[0], pos[1], pos[2], 1, 1]))


def fitgaussian_1d_fix_mu(data, x, pos):
    x, data = x.ravel(), data.ravel()
    x = x[data > _threshold]
    data = data[data > _threshold]

    def _error_function(p):
        return np.ravel(_gaussian_1d_fix_mu(*p)(x - pos) - data)

    return opt.leastsq(_error_function, np.array([1, 1]))


def fitgaussian_2d_fix_mu(data, x, y, pos):
    x, y, data = x.ravel(), y.ravel(), data.ravel()
    x = x[data > _threshold]
    y = y[data > _threshold]
    data = data[data > _threshold]

    def _error_function(p):
        return np.ravel(_gaussian_2d_fix_mu(*p)(x - pos[0], y - pos[1]) - data)

    return opt.leastsq(_error_function, np.array([1, 1, 1]))


def fitgaussian_3d_fix_mu(data, x, y, z, pos):
    x, y, z, data = x.ravel(), y.ravel(), z.ravel(), data.ravel()
    x = x[data > _threshold]
    y = y[data > _threshold]
    z = z[data > _threshold]
    data = data[data > _threshold]

    def _error_function(p):
        return np.ravel(_gaussian_3d_fix_mu(*p)(x - pos[0], y - pos[1], z - pos[2]) - data)

    return opt.leastsq(_error_function, np.array([1, 1, 1]))
