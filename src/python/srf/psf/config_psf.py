#!/usr/bin/env python
# encoding: utf-8
'''
@author: Minghao Guo, Xiang Hong, Gaoyu Chen and Weijie Tao
@license: LGPL_v3.0
@contact: mh.guo0111@gmail.com
@software: srf_v2
@file: con_psf.py
@date: 11/16/2018
@desc: new version of Scalable Reconstruction Framework for Medical Imaging
'''

import numpy as np

from srf.psf.data.image import *
from srf.psf.data.psf import *
from srf.psf.meta.image_meta import *
from srf.psf.meta.psf_meta import *

_sqrt_pi = np.sqrt(np.pi)


def config_psf(config):
    meta = PsfMeta3d()
    shape = tuple(config['output']['image']['grid'])
    center = tuple(config['output']['image']['center'])
    size = tuple(config['output']['image']['size'])
    image_meta = Image_meta_3d(shape, center, size)

    psf = config['psf']
    for key, val in psf.items():
        print('fitting...', key)

        image_data = np.load(val['path_file'])
        image = Image_3d(image_data, image_meta)
        if val['category'] == 'xy':
            meta.add_para_xy(image, tuple(val['pos']), 20)
        else:
            meta.add_para_z(image, tuple(val['pos']), 20)

    psf = PSF_3d(meta, image_meta)
    psf.generate_matrix_all()
    psf.save_h5('mat_psf.h5')


if __name__ == "__main__":
    import json

    with open('API2.json', 'r') as fin:
        config = json.load(fin)
        config_psf(config)
