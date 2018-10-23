from dxl.learn import Model
from srf.data import Image
from scipy.optimize import curve_fit
from scipy import asarray as ar, exp
import numpy as np
__all__ = ['PSFStep']

EPS = 1e-4
class PSFStep:
    class KEYS:
        class PARA:
            POS = 'pos'
            POS0 = 'pos0'
            Axy = 'a_xy'
            Az = 'a_z'
            SIGMA_x = 'sigma_x'
            SIGMA_y = 'sigma_y'
            SIGMA_z = 'sigma_z'

    def __init__(self, name, img0, img1):
        self.current_image = img0
        self.corrected_image = img1

    def _cal_diff_pos(self, ind):
        # pos_all = self.KEYS.PARA.POS[ind]
        pos0_all = self.KEYS.PARA.POS0
        l_pos = ind.shape[0]
        l_pos0 = pos0_all.shape[0]
        results = np.array((l_pos, l_pos0, 3), dtype=int)
        for i in range(ind.shape[0]):
            results[i] = ind[i, :] - pos0_all
        return results

    def _search_pos(self):
        (nx, ny, nz) = np.where(self.corrected_image > EPS)
        return np.vstack((nx, ny, nz)).T

    def fitting(self):
        ind = self._search_pos()
        pos_diff = self._cal_diff_pos(ind)
        img_diff = self.current_image[ind]
        popt, pcov = curve_fit(_gaus, pos_diff, img_diff)
        self.KEYS.PARA.Axy = popt[0]
        self.KEYS.PARA.Az = popt[1]
        self.KEYS.PARA.SIGMA_x = popt[2]
        self.KEYS.PARA.SIGMA_y = popt[3]
        self.KEYS.PARA.SIGMA_z = popt[4]


def _gaus(pos_diff, axy, az, sigma_x, sigma_y, sigma_z):
    value = np.zeros(1)
    for j in range(len(pos_diff)):
        value = value + axy[j] * az[j]\
                * exp(-pos_diff[j][0] ** 2 / (2 * sigma_x[j] ** 2)) \
                * exp(-pos_diff[j][1] ** 2 / (2 * sigma_y[j] ** 2)) \
                * exp(-pos_diff[j][2] ** 2 / (2 * sigma_z[j] ** 2))
    return value

