from doufo import List, Pair, x
from doufo.collections import DataArray
from dxl.shape.data import Point
from srf.data import PositionEvent, LoR, PETSinogram3D, PETCylindricalScanner
import numpy as np

__all__ = ['ndarray2listmode']


def ndarray2listmode(data: np.ndarray) -> List[LoR]:
    if is_positional_data(data):
        return ndarray2listmode_positional(data)
    raise ValueError(f"Can't convert input ndarray to ListModeData")


def ndarray2listmode_positional(data: np.ndarray) -> List[LoR]:
    return List([parse_positional_event(r) for r in data])


def parse_positional_event(row) -> LoR:
    return LoR(PositionEvent(Point(row[:3])),
               PositionEvent(Point(row[3:6])))


def is_positional_data(data: np.ndarray) -> bool:
    # FIXME add detections
    return True
