'''
This file provides the methods to translate SRF format list-mode data to CASTOR format data.
CASTOR format data consists of an ascii .cdh header file and a binary data file.
'''
from doufo import List, Pair, x
from dxl.shape.data import Point
from srf.data import DetectorIdEvent, LoR, PETCylindricalScanner, PositionEvent, DetectorIdEvent
# from .on_event import position2detectorid
import numpy as np
from functools import partial
from srf.external.castor.io import DataHeader, DataListModeEvent
from .geom_calculation import compute_crystal_id, compute_ring_id




__all__ = ['listmode2cdhf']
def listmode2cdhf(config_file, scanner: PETCylindricalScanner, listmode_data: List[LoR]):
    config = 
    cdh = generate_cdh()
    cdf = generate_cdf()
    return cdh, cdf 

def generate_cdh():
    
    pass

def generate_cdf():
    pass