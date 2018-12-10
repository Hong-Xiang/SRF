import numpy as np

__all__ = ['TOF', 'PSF', 'Scattering', 'Attenuation', 'Correction', 'make_correction']

class TOF():
    def __init__(self, res:np.float32, bin: np.float32):
        self._resolution = res
        self._bin = bin
    
    @property
    def resolution(self):
        return self._resolution
    
    @property
    def bin(self):
        return self._bin

class PSF():
    def __init__(self, flag:bool, xy_file: str, z_file:str):
        self._flag = flag
        self._xy = xy_file
        self._z = z_file
    
    @property
    def flag(self):
        return self._flag
    @property
    def xy(self):
        return self._xy
    
    @property
    def z(self):
        return self._z

class Attenuation():
    def __init__(self, map_file: str):
        self._map_file = map_file
    @property
    def map_file(self):
        return self._map_file

class Scattering():
    def __init__(self):
        pass

class Correction:
    def __init__(self, psf:PSF , attenuation: Attenuation, scattering: Scattering):
        self._psf = psf
        self._attenuation = attenuation
        self._scattering = scattering
    
    @property 
    def psf(self):
        return self._psf
    
    @property
    def attenuation(self):
        return self._attenuation
    
    @property
    def scattering(self):
        return self._scattering


def make_correction(correction_config:dict):
    if not isinstance(correction_config, dict):
        raise TypeError("The correction_config is not a dict!")
    else:
        cc = correction_config
        if cc.__contains__('psf_kernel'):
            if cc['psf_kernel']['flag'] is True:
                print(cc['psf_kernel']['flag'])
                psf = PSF(cc['psf_kernel']['flag'], cc['psf_kernel']['psf_xy'], cc['psf_kernel']['psf_z'])
            else:
                psf = None
        else:
            psf = None
        if cc.__contains__('atten_correction'):
            attenuation = Attenuation(map_file = cc['atten_correction']['map_file'])
        else:
            attenuation = None
        if cc.__contains__('scattering'):
            scattering = Scattering()
        else:
            scattering = None
        # print(psf)
        # print(attenuation)
        # print(scattering)

        return Correction(psf = psf, attenuation = attenuation, scattering = scattering)