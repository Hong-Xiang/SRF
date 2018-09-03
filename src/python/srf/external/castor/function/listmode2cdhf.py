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
from srf.external.castor.data import DataHeader, DataListModeEvent, DataHeaderScript
# from .geom_calculation import compute_crystal_id, compute_ring_id

from srf.external.castor.io import save_cdh, render

__all__ = ['listmode2cdhf']
def listmode2cdhf(config:dict):
    header = DataHeader(config)
    path, data = generate_cdh(header)
    print('generate!')
    raw_data = config['input_data_file'] 
    save_cdh(path, data)


def generate_cdh(header):
    header_script = DataHeaderScript(spec = header)
    file_name = header.data_file_name
    data_str = render(header_script)
    return file_name, data_str

def generate_cdf():
    pass
