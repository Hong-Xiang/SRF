import h5py
import numpy as np
from srf.data import LoR, PositionEvent, DetectorIdEvent
from typing import Dict

__all__ = []

DEFAULT_GROUP_NAME = 'listmode_data'

DEFAULT_COLUMNS = ['fst', 'snd', 'weight', 'tof']

# TODO move implementation to doufo.io

def load_h5(path, group_name=DEFAULT_GROUP_NAME)-> Dict[str, np.ndarray]:
    with h5py.File(path, 'r') as fin:
        dataset = fin[group_name]
        result = {}
        for k in DEFAULT_COLUMNS:
            if k in dataset:
                result[k] = np.array(dataset[k])
        return result


def save_h5(path, dct, group_name=DEFAULT_GROUP_NAME):
    with h5py.File(path, 'w') as fout:
        group = fout.create_group(group_name)
        for k, v in dct.items():
            group.create_dataset(k, data=v, compression="gzip")

