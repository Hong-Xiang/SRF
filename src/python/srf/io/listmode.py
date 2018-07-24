import h5py
import numpy as np
from srf.data import LoR, PositionEvent, DetectorIdEvent, ListModeData


def load_listmode_data(path):
    with h5py.File(path, 'r') as fin:
        dataset = fin['dataset']
        if dataset.shape[1] == 6:
            if dataset.dtype == np.float32:
                return ListModeData([LoR(PositionEvent(d[:3]),
                                         PositionEvent(d[3:6]))
                                     for d in dataset])
            if dataset.dtype == np.int32:
                return ListModeData([LoR(DetectorIdEvent(d[0], d[1], d[2]),
                                         DetectorIdEvent(d[3], d[4], d[5]))
                                     for d in dataset])
