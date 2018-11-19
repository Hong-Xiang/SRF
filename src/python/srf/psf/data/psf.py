#!/usr/bin/env python
# encoding: utf-8
'''
@author: Minghao Guo, Xiang Hong, Gaoyu Chen and Weijie Tao
@license: LGPL_v3.0
@contact: mh.guo0111@gmail.com
@software: srf_v2
@file: psf.py
@date: 11/13/2018
@desc: new version of Scalable Reconstruction Framework for Medical Imaging
'''

import h5py
import numpy as np
import scipy.interpolate as interp
from scipy import sparse
from srf2.meta.image_meta import Image_meta_3d
from srf2.meta.psf_meta import PsfMeta3d
from tqdm import tqdm

__all__ = ('PSF_3d',)
_sqrt_2_pi = np.sqrt(np.pi * 2)
eps = 1e-8


class PSF_3d:
    def __init__(self, meta: PsfMeta3d = PsfMeta3d(), image_meta: Image_meta_3d =
    Image_meta_3d(), matrix_xy = None, matrix_z = None, matrix_xy_full = None, matrix_z_full =
                 None):
        self._meta = meta
        self._image_meta = image_meta
        self._matrix_xy = matrix_xy
        self._matrix_z = matrix_z
        self._matrix_xy_full = matrix_xy_full
        self._matrix_z_full = matrix_z_full

    @property
    def meta(self):
        return self._meta

    @property
    def image_meta(self):
        return self._image_meta

    @property
    def matrix_xy(self):
        return self._matrix_xy

    @property
    def matrix_z(self):
        return self._matrix_z

    @property
    def matrix_xy_full(self):
        return self._matrix_xy_full

    @property
    def matrix_z_full(self):
        return self._matrix_z_full

    def save_h5(self, path = None, mode = 'w'):
        if path is None:
            path = 'tmp' + self.__class__.__name__ + '.h5'
        self.meta.save_h5(path, mode)
        self.image_meta.save_h5(path, 'r+')
        with h5py.File(path, 'r+') as fout:
            group = fout.create_group('PSF')
            if self._matrix_xy is not None:
                row, col = self.matrix_xy.nonzero()
                data = self.matrix_xy.data
                group.create_dataset('_matrix_xy_row', data = row, compression = "gzip")
                group.create_dataset('_matrix_xy_col', data = col, compression = "gzip")
                group.create_dataset('_matrix_xy_data', data = data, compression = "gzip")

            if self._matrix_z is not None:
                row, col = self.matrix_z.nonzero()
                data = self.matrix_z.data
                group.create_dataset('_matrix_z_row', data = row, compression = "gzip")
                group.create_dataset('_matrix_z_col', data = col, compression = "gzip")
                group.create_dataset('_matrix_z_data', data = data, compression = "gzip")

            if self._matrix_xy_full is not None:
                row, col = self.matrix_xy_full.nonzero()
                data = self.matrix_xy_full.data
                group.create_dataset('_matrix_xy_full_row', data = row, compression = "gzip")
                group.create_dataset('_matrix_xy_full_col', data = col, compression = "gzip")
                group.create_dataset('_matrix_xy_full_data', data = data, compression = "gzip")

            if self._matrix_z_full is not None:
                row, col = self.matrix_z_full.nonzero()
                data = self.matrix_z_full.data
                group.create_dataset('_matrix_z_full_row', data = row, compression = "gzip")
                group.create_dataset('_matrix_z_full_col', data = col, compression = "gzip")
                group.create_dataset('_matrix_z_full_data', data = data, compression = "gzip")

    @classmethod
    def load_h5(cls, path = None, name = ''):
        if path is None:
            path = 'tmp' + cls.__name__ + '.h5'

        meta = PsfMeta3d.load_h5(path)
        image_meta = Image_meta_3d.load_h5(path)
        with h5py.File(path, 'r') as fin:
            dataset = fin['PSF']
            if name == 'meta':
                return PSF_3d(meta, image_meta)

            row = np.array(dataset['_matrix_xy_row'])
            col = np.array(dataset['_matrix_xy_col'])
            data = np.array(dataset['_matrix_xy_data'])
            matrix_xy = sparse.csr_matrix((data, (row, col)),
                                          shape = (image_meta.n_xy, image_meta.n_xy),
                                          dtype = np.float32)

            row = np.array(dataset['_matrix_z_row'])
            col = np.array(dataset['_matrix_z_col'])
            data = np.array(dataset['_matrix_z_data'])
            matrix_z = sparse.csr_matrix((data, (row, col)),
                                         shape = (image_meta.n_z, image_meta.n_z),
                                         dtype = np.float32)
            if name == 'all':
                row = np.array(dataset['_matrix_xy_full_row'])
                col = np.array(dataset['_matrix_xy_full_col'])
                data = np.array(dataset['_matrix_xy_full_data'])
                matrix_xy_full = sparse.csr_matrix((data, (row, col)),
                                                   shape = (image_meta.n_all, image_meta.n_all),
                                                   dtype = np.float32)

                row = np.array(dataset['_matrix_z_full_row'])
                col = np.array(dataset['_matrix_z_full_col'])
                data = np.array(dataset['_matrix_z_full_data'])
                matrix_z_full = sparse.csr_matrix((data, (row, col)),
                                                  shape = (image_meta.n_all, image_meta.n_all),
                                                  dtype = np.float32)
                return PSF_3d(meta, image_meta, matrix_xy, matrix_z, matrix_xy_full, matrix_z_full)

            return PSF_3d(meta, image_meta, matrix_xy, matrix_z)

    def generate_matrix_xy_full(self):
        lil_xy = sparse.lil_matrix((self.image_meta.n_all, self.image_meta.n_all),
                                   dtype = np.float32)
        row, col = self.matrix_xy.nonzero()
        data = self.matrix_xy.data
        print('Generating full PSF matrix xy')
        for iz in tqdm(np.arange(self.image_meta.n_z)):
            lil_xy[row * self.image_meta.n_z + iz, col * self.image_meta.n_z + iz] = data
        self._matrix_xy_full = lil_xy.tocsr()
        return self.matrix_xy_full

    def generate_matrix_z_full(self):
        lil_z = sparse.lil_matrix((self.image_meta.n_all, self.image_meta.n_all),
                                  dtype = np.float32)
        row, col = self.matrix_z.nonzero()
        data = self.matrix_z.data
        print('Generating full PSF matrix z')
        for ix in tqdm(np.arange(self.image_meta.n_x)):
            for iy in np.arange(self.image_meta.n_y):
                ind = iy + ix * self.image_meta.n_y
                lil_z[row + self.image_meta.n_z * ind, col + self.image_meta.n_z * ind] = data
        self._matrix_z_full = lil_z.tocsr()
        return self.matrix_z_full

    def generate_matrix_all(self):
        print('Generating all PSF matrix')
        self.generate_matrix_xy()
        self.generate_matrix_z()
        # self.generate_matrix_xy_full()
        # self.generate_matrix_z_full()

    def generate_matrix_xy(self):
        x1, y1 = self.image_meta.grid_centers_2d()

        R1 = np.sqrt(x1 ** 2 + y1 ** 2)
        R0 = np.sqrt(self.meta.mu[:, 0] ** 2 + self.meta.mu[:, 1] ** 2)
        lil_matrix_xy = sparse.lil_matrix((self.image_meta.n_xy, self.image_meta.n_xy),
                                          dtype = np.float32)
        ind_xy = np.where(self.meta.sigma[:, 0] != 0)[0]
        if ind_xy.size == 0:
            lil_matrix_xy[range(self.image_meta.n_xy), range(self.image_meta.n_xy)] = 1
            self._matrix_xy = lil_matrix_xy.tocsr()
            return self.matrix_xy
        elif ind_xy.size == 1:
            sigma_x = self.meta.sigma[ind_xy, 0] * np.ones(R1.shape)
            sigma_y = self.meta.sigma[ind_xy, 1] * np.ones(R1.shape)
        else:
            fsigx = interp.interp1d(R0[ind_xy], self.meta.sigma[ind_xy, 0], kind = 'quadratic',
                                    fill_value = 'extrapolate')
            fsigy = interp.interp1d(R0[ind_xy], self.meta.sigma[ind_xy, 1], kind = 'quadratic',
                                    fill_value = 'extrapolate')
            sigma_x, sigma_y = fsigx(R1), fsigy(R1)
        print('Generating psf matrix in xy')
        theta = self.image_meta.theta()
        for ix in tqdm(np.arange(self.image_meta.n_x)):
            for iy in np.arange(self.image_meta.n_y):
                ind = iy + ix * self.image_meta.n_y
                if R1[ix, iy] < np.max(R0):
                    img_tmp = _gaussian_2d((x1 - x1[ix, iy], y1 - y1[ix, iy], theta[ix, iy]),
                                           sigma_x[ix, iy], sigma_y[ix, iy]) * \
                              self.image_meta.unit_size[0] * self.image_meta.unit_size[1]
                    gk = img_tmp.flatten()
                    row = np.where(gk > eps)[0]
                    col = ind * np.ones(row.size)
                    data = gk[row]
                else:
                    row = col = ind * np.ones([1])
                    data = np.ones([1])
                # col = np.where(gk > eps)[0]
                # row = ind * np.ones(col.shape)
                # data = gk[col]
                lil_matrix_xy[row, col] = data

        self._matrix_xy = lil_matrix_xy.tocsr()
        return self.matrix_xy

    def generate_matrix_z(self):
        # z1 = self.image_meta.meshgrid_1d()
        z1 = self.image_meta.grid_centers_1d()
        z0 = np.abs(self.meta.mu[:, 2])

        lil_matrix_z = sparse.lil_matrix((self.image_meta.n_z, self.image_meta.n_z),
                                         dtype = np.float32)
        ind_z = np.where(self.meta.sigma[:, 2] != 0)[0]
        if ind_z.size == 0:
            lil_matrix_z[range(self.image_meta.n_z), range(self.image_meta.n_z)] = 1
            self._matrix_xy = lil_matrix_z.tocsr()
            return self.matrix_z
        elif ind_z.size == 1:
            sigma_z = self.meta.sigma[ind_z, 2] * np.ones(z1.shape)
        else:
            fsigz = interp.interp1d(z0[ind_z], self.meta.sigma[ind_z, 2], kind = 'quadratic',
                                    fill_value = 'extrapolate')
            sigma_z = fsigz(np.abs(z1))

        print('Generating psf matrix in z')
        for iz in tqdm(np.arange(self.image_meta.n_z)):
            if z1[iz] < np.max(z0):
                img_tmp = _gaussian_1d(z1 - z1[iz], sigma_z[iz]) * self.image_meta.unit_size[2]
                gk = img_tmp.flatten()
                # row = np.where(gk > eps)[0]
                # col = iz * np.ones(row.size)
                # data = gk[row]
                col = np.where(gk > eps)[0]
                row = iz * np.ones(col.shape)
                data = gk[col]
            else:
                row = col = iz * np.ones([1])
                data = np.ones([1])
            lil_matrix_z[row, col] = data
        self._matrix_z = lil_matrix_z.tocsr()
        return self.matrix_z


class PSF_2d:
    pass


def _gaussian_1d(z, sigz):
    sigz = abs(sigz)
    return 1 / _sqrt_2_pi / sigz * np.exp(-z ** 2 / 2 / sigz ** 2)


def _gaussian_2d(x_y_t, sigx, sigy):
    x = x_y_t[0]
    y = x_y_t[1]
    theta = x_y_t[2]
    sigx = abs(sigx)
    sigy = abs(sigy)
    x1 = x * np.cos(theta) + y * np.sin(theta)
    y1 = -x * np.sin(theta) + y * np.cos(theta)
    return 1 / _sqrt_2_pi ** 2 / sigx / sigy * np.exp(-x1 ** 2 / 2 / sigx ** 2) * np.exp(
        - y1 ** 2 / 2 / sigy ** 2)


def _gaussian_3d_xy(x_y_z_t, sigx, sigy):
    x = x_y_z_t[0]
    y = x_y_z_t[1]
    z = x_y_z_t[2]
    t = x_y_z_t[3]
    sigx = abs(sigx)
    sigy = abs(sigy)

    x1 = x * np.cos(t) + y * np.sin(t)
    y1 = -x * np.sin(t) + y * np.cos(t)

    return 1 / _sqrt_2_pi ** 2 / sigx / sigy * np.exp(-x1 ** 2 / 2 / sigx ** 2) * np.exp(
        - y1 ** 2 / 2 / sigy ** 2)


def _gaussian_3d(x_y_z_t, sigx, sigy, sigz):
    x = x_y_z_t[0]
    y = x_y_z_t[1]
    z = x_y_z_t[2]
    t = x_y_z_t[3]

    sigx = abs(sigx)
    sigy = abs(sigy)
    sigz = abs(sigz)

    x1 = x * np.cos(t) + y * np.sin(t)
    y1 = -x * np.sin(t) + y * np.cos(t)
    return 1 / _sqrt_2_pi ** 3 / sigx / sigy / sigz * np.exp(-x1 ** 2 / 2 / sigx ** 2) * np.exp(
        - y1 ** 2 / 2 / sigy ** 2) * np.exp(-z ** 2 / 2 / sigz ** 2)
