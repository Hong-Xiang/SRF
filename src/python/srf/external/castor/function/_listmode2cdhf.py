from doufo import List, Pair, x
from dxl.shape.data import Point
from srf.data import DetectorIdEvent, LoR, PETSinogram3D, PETCylindricalScanner, PositionEvent, DetectorIdEvent
# from .on_event import position2detectorid
import numpy as np
from functools import partial

__all__ = ['listmode2cdhf']
def listmode2cdhf(scanner: PETCylindricalScanner, listmode_data: List[LoR]):
    cdh = []
    cdf = []
    return cdh, cdf
