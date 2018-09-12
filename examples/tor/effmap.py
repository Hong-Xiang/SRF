import time
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from srf.function import make_scanner
#from srf.scanner.pet import CylindricalPET,RingGeometry,TOF,Block, RingBlock
from srf.preprocess.function.on_tor_lors import map_process

from srf.graph.reconstruction.ring_efficiency_map import RingEfficiencyMap

from srf.model import BackProjectionOrdinary
from srf.physics import SplitLoRsModel
from srf.preprocess import merge_effmap
from dxl.learn.session import Session
from srf.data import ListModeDataWithoutTOF, ListModeDataSplitWithoutTOF
import tensorflow as tf
from srf.function import create_listmode_data


def compute(lors, grid, center, size):
    physics_model = SplitLoRsModel('map_model')
    backprojection_model = BackProjectionOrdinary(physics_model)
    t = RingEfficiencyMap('effmap', backprojection_model, lors, grid=grid, center=center, size=size)
    t.make()
    with Session() as sess:
        result = t.run()
        print(result[result > 0])
    sess.reset()
    return result


def get_mct_config():
    """
    Retrun a dict that describes the geometry of mCT scanner.
    """
    config = {
        "modality": "PET",
        "name": "mCT",
        "ring": {
            "inner_radius": 400,
            "outer_radius": 420,
            "axial_length": 1419.3,
            "nb_rings": 415,
            "nb_blocks_per_ring": 48,
            "gap": 0.0
        },
        "block": {
            "grid": [1, 15, 1],
            "size": [20.0, 51.3, 3.42],
            "interval": [0.0, 0.0, 0.0]
        },
        "tof": {
            "resolution": 530,
            "bin": 40
        }
    }
    return config


def main():
    grid = [195, 195, 415]
    center = [0.0, 0.0, 0.0]
    size = [666.9, 666.9, 1419.3]
    config = get_mct_config()
    rpet = make_scanner('Cylinder',config)
    r1 = rpet.rings[0]

    for ir in tqdm(range(0, rpet.nb_rings)):
        print("start to compute the {} th map.".format(ir))
        r2 = rpet.rings[ir]
        lors = rpet.make_ring_pairs_lors(r1, r2)
        projection_data = create_listmode_data[ListModeDataSplitWithoutTOF](lors)
        result = compute(projection_data, grid, center, size)

        np.save('effmap_{}.npy'.format(ir), result)

    merge_effmap(0, rpet.nb_rings, rpet.nb_rings, 1, './')


if __name__ == '__main__':
    main()
