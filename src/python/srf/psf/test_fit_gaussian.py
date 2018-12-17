# encoding: utf-8
'''
@author: Minghao Guo, Xiang Hong, Gaoyu Chen and Weijie Tao
@license: LGPL_v3.0
@contact: mh.guo0111@gmail.com
@software: srf_v2
@file: test_fit_gaussian.py
@date: 12/17/2018
@desc: new version of Scalable Reconstruction Framework for Medical Imaging
'''
from .gaussian_fit import fit_gaussian
import numpy as np

class TestFit_gaussian(TestCase):
    def test_fit_gaussian_1d(self):
        x = np.linspace(-1, 1, 10)
        sigx2, mux = 3, 0.1
        g = np.exp(-(x - mux) ** 2 / 2 / sigx2)
        p1 = fit_gaussian(g, x, mode = 'fix_mu', mu = 0.1)
        p2 = fit_gaussian(g, x, mode = 'fit_mu')
        p3 = fit_gaussian(g, x, mode = 'fit_mu', mu = 0.1)

    def test_fit_gaussian_2d(self):
        x, y = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
        sigx2, sigy2, mux, muy = 3, 2, 0.2, 0.1
        g = np.exp(-(x - mux) ** 2 / 2 / sigx2) * np.exp(-(y - muy) ** 2 / 2 / sigy2)
        p1 = fit_gaussian(g, (x, y), mode = 'fix_mu', mu = (0.2, 0.1))
        p2 = fit_gaussian(g, (x, y), mode = 'fit_mu')
        p3 = fit_gaussian(g, (x, y), mode = 'fit_mu', mu = (0.2, 0.1))
