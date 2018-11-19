#!/usr/bin/env python
# encoding: utf-8
'''
@author: Minghao Guo, Xiang Hong, Gaoyu Chen and Weijie Tao
@license: LGPL_v3.0
@contact: mh.guo0111@gmail.com
@software: srf_v2
@file: projection_meta.py
@date: 11/11/2018
@desc: new version of Scalable Reconstruction Framework for Medical Imaging
'''

import attr
import numpy as np

from srf2.core.abstracts import *

__all__ = ('Projection_meta', 'Listmode_projection_meta', 'Sinogram_projection_meta',
           'Projection_meta_singleton', 'Listmode_projection_meta_singleton',
           'Sinogram_projection_meta_singleton',)


@attr.s
class Projection_meta(Meta):
    _shape = attr.ib(default = (1,) * 3)
    _center = attr.ib(default = (0,) * 3)
    _size = attr.ib(default = (1,) * 3)
    _inner_radius = attr.ib(default = 99.0)
    _outer_radius = attr.ib(default = 119.0)
    _axial_length = attr.ib(default = 33.4)
    _nb_rings = attr.ib(default = 1, converter = int)
    _nb_blocks_per_ring = attr.ib(default = 16, converter = int)
    _gap = attr.ib(default = 0.0)

    @property
    def shape(self):
        return self._shape

    @property
    def center(self):
        return self._center

    @property
    def size(self):
        return self._size

    @property
    def unit_size(self):
        return tuple(ti / tj for (ti, tj) in zip(self.size, self.shape))

    @property
    def inner_radius(self) -> float:
        return self._inner_radius

    @property
    def outer_radius(self) -> float:
        return self._outer_radius

    @property
    def axial_length(self) -> float:
        return self._axial_length

    @property
    def nb_rings(self):
        return self._nb_rings

    @property
    def nb_blocks_per_ring(self):
        return self._nb_blocks_per_ring

    @property
    def gap(self):
        return self._gap

    @property
    def n_sensors_per_block(self):
        return np.prod(self._shape)

    @property
    def n_sensors_per_ring(self):
        return self.n_sensors_per_block * self.nb_blocks_per_ring

    @property
    def n_sensors_all(self):
        return self.n_sensors_per_ring * self.nb_rings

    def fmap(self, f):
        pass

    def transfer(self, data_type = None):
        raise NotImplementedError


class Listmode_projection_meta(Projection_meta):
    def __init__(self, *argv):
        super().__init__(*argv)

    def transfer(self, data_type = None):
        if data_type == 'sinogram':
            return Sinogram_projection_meta(self.shape, self.center, self.size, self.inner_radius,
                                            self.outer_radius,
                                            self.axial_length, self.nb_rings,
                                            self.nb_blocks_per_ring, self.gap)
        elif data_type == 'listmode':
            return self
        else:
            raise NotImplementedError


class Sinogram_projection_meta(Projection_meta):
    def __init__(self, *argv):
        super().__init__(*argv)

    def transfer(self, data_type = None):
        if data_type == 'listmode':
            return Listmode_projection_meta(self.shape, self.center, self.size, self.inner_radius,
                                            self.outer_radius,
                                            self.axial_length, self.nb_rings,
                                            self.nb_blocks_per_ring, self.gap)
        elif data_type == 'sinogram':
            return self
        else:
            raise NotImplementedError


class Projection_meta_singleton(Projection_meta, Singleton):
    pass


class Sinogram_projection_meta_singleton(Sinogram_projection_meta, Singleton):
    pass


class Listmode_projection_meta_singleton(Listmode_projection_meta, Singleton):
    pass
