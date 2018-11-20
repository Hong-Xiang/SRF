#!/usr/bin/env python
# encoding: utf-8
'''
@author: Minghao Guo, Xiang Hong, Gaoyu Chen and Weijie Tao
@license: LGPL_v3.0
@contact: mh.guo0111@gmail.com
@software: srf_v2
@file: psf_meta.py
@date: 11/11/2018
@desc: new version of Scalable Reconstruction Framework for Medical Imaging
'''

import numpy as np
import scipy.optimize as opt

from srf.psf.core.abstracts import Meta

__all__ = ('PsfMeta3d', 'PsfMeta2d',)

_sqrt_2_pi = np.sqrt(np.pi * 2)


class PsfMeta3d(Meta):
    def __init__(self, amp = tuple([]), mu = tuple([]), sigma = tuple([])):
        self._amp = np.array(amp, dtype = np.float32)
        self._mu = np.array(mu, dtype = np.float32)
        self._sigma = np.array(sigma, dtype = np.float32)

    @property
    def sigma(self):
        return self._sigma

    @property
    def mu(self):
        return self._mu

    @property
    def amp(self):
        return self._amp

    def add_para_xy(self, image, pos = (0, 0, 0), rang = 20):
        image = image.transpose(('x', 'y', 'z'))
        ix1, iy1, iz1 = image.meta.locate(pos)
        slice_x = slice_y = slice(None, None)
        slice_z = slice(int(np.round(iz1) - rang / image.meta.unit_size[2]),
                        int(np.round(iz1) + rang / image.meta.unit_size[2]) + 1)

        x1, y1 = image.meta.grid_centers_2d([slice_x, slice_y, slice_z])
        x1, y1 = x1 - pos[0], y1 - pos[1]
        x1 /= image.meta.unit_size[0]
        y1 /= image.meta.unit_size[1]
        image_new_data = np.sum(image[slice_x, slice_y, slice_z].normalize().data, axis = 2)

        p = _fitgaussian_2d(image_new_data, x1, y1)

        out_amp = np.array([p[0][0]])
        out_sigma = np.abs(np.array([p[0][1], p[0][2], 0]))
        if self._mu.size == 0:
            self._mu = np.array(pos)
            self._sigma = np.array(out_sigma)
            self._amp = np.array(out_amp)
        else:
            self._mu = np.vstack((self._mu, [pos]))
            self._sigma = np.vstack((self._sigma, out_sigma))
            self._amp = np.hstack((self._amp, out_amp))
        return x1, y1, image_new_data, p[0]

    def add_para_z(self, image, pos = (0, 0, 0), rang = 20):
        image = image.transpose(('x', 'y', 'z'))
        ix1, iy1, iz1 = image.meta.locate(pos)
        slice_x = slice(int(np.round(ix1) - rang / image.meta.unit_size[0]),
                        int(np.round(ix1) + rang / image.meta.unit_size[0]) + 1)
        slice_y = slice(int(np.round(iy1) - rang / image.meta.unit_size[1]),
                        int(np.round(iy1) + rang / image.meta.unit_size[1]) + 1)
        slice_z = slice(None, None)

        z1 = image.meta.grid_centers_1d([slice_x, slice_y, slice_z]) - pos[2]
        z1 /= image.meta.unit_size[2]
        image_new_data = np.sum(image[slice_x, slice_y, slice_z].normalize().data, axis = (0, 1))
        p = _fitgaussian_1d(image_new_data, z1)

        out_amp = np.array([p[0][0]])
        out_sigma = np.abs(np.array([0, 0, p[0][1]]))

        if self._mu.size == 0:
            self._mu = np.array(pos)
            self._sigma = np.array(out_sigma)
            self._amp = np.array(out_amp)
        else:
            self._mu = np.vstack((self._mu, [pos]))
            self._sigma = np.vstack((self._sigma, out_sigma))
            self._amp = np.hstack((self._amp, out_amp))
        return z1, image_new_data, p[0]


class PsfMeta2d(Meta):
    pass


def _gaussian_1d(amp, sigz):
    sigz = abs(sigz)
    return lambda z: amp * np.exp(-z ** 2 / 2 / sigz ** 2)


def _gaussian_2d(amp, sigx, sigy):
    sigx = abs(sigx)
    sigy = abs(sigy)
    return lambda x, y: amp * np.exp(-x ** 2 / 2 / sigx ** 2) * np.exp(-y ** 2 / 2 / sigy ** 2)


def _fitgaussian_2d(data, x, y):
    def _error_function(p):
        return np.ravel(_gaussian_2d(*p)(x, y) - data)

    return opt.leastsq(_error_function, np.array([1, 1, 1]))


def _fitgaussian_1d(data, x):
    def _error_function(p):
        return np.ravel(_gaussian_1d(*p)(x) - data)

    return opt.leastsq(_error_function, np.array([1, 1]))
