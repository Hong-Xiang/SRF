# encoding: utf-8
'''
@author: Minghao Guo, Xiang Hong, Gaoyu Chen and Weijie Tao
@license: LGPL_v3.0
@contact: mh.guo0111@gmail.com
@software: srf_v2
@file: psf_main.py
@date: 12/17/2018
@desc: new version of Scalable Reconstruction Framework for Medical Imaging
'''
import json
import numpy as np
from srf.psf.gaussian_fit import fit_gaussian

def main(config):
    out_path = config['psf']['out_path']
    psf_xy = config['psf']['psf_xy']
    psf_z = config['psf']['psf_z']
    n_slice = 3

    # psf xy fitting
    n_xy = len(psf_xy['pos'])
    amp = np.zeros(n_xy,)
    sigx = np.zeros(n_xy,)
    sigy = np.zeros(n_xy,)
    mux = np.zeros(n_xy,)
    muy = np.zeros(n_xy,)
    for i in range(n_xy):
        path = psf_xy['path_prefix'] + psf_xy['ind'][i] + psf_xy['path_postfix']
        img = np.load(path)
        pos = psf_xy['pos'][i]
        img_sum = np.average(img[:, :, pos[2] - n_slice: pos[2] + n_slice])
        [y, x] = np.meshgrid(np.arange(img_sum.shape[1]), np.arange(img_sum.shape[0]))
        amp[i], mux[i], muy[i], sigx[i], sigy[i] = fit_gaussian(img_sum, (x, y),  mode = 'fit_mu', mu = pos)
    output = np.vstack((amp, sigx, sigy, mux, muy))
    np.save(out_path, output)
if __name__ == "__main__":
    import json

    with open('API_psf.json', 'r') as fin:
        config = json.load(fin)
        main(config)
