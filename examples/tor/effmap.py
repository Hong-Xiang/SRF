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
from srf.preprocess.preprocess import preprocess
from dxl.learn.session import Session
from srf.data import ScannerClass,ListModeDataWithoutTOF, ListModeDataSplitWithoutTOF
import tensorflow as tf
from srf.function import create_listmode_data


def compute(lors, grid, center, size, config):
    physics_model = SplitLoRsModel(config['projection_model']['kernel_width'],'map_model')
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
            "nb_ring": 415,
            "nb_block_per_ring": 48,
            "gap": 0.0
        },
        "block": {
            "grid": [1, 15, 1],
            "size": [20.0, 51.3, 3.42],
            "interval": [0.0, 0.0, 0.0]
        },
        'projection_model': {
            'tof_sigma2': 162.30,
            'tof_bin': 6.0,
            'kernel_width': 3.86,
        }
    }
    return config


def main():
    grid = [195, 195, 415]
    center = [0.0, 0.0, 0.0]
    size = [666.9, 666.9, 1419.3]
    config = get_mct_config()
    rpet = make_scanner(ScannerClass.CylinderPET,config)
    r1 = rpet.rings[0]

    for ir in tqdm(range(0, rpet.nb_rings)):
        print("start to compute the {} th map.".format(ir))
        r2 = rpet.rings[ir]
        lors = rpet.make_ring_pairs_lors(r1, r2)
        lors = preprocess(lors)
        projection_data = create_listmode_data[ListModeDataSplitWithoutTOF](lors)
        result = compute(projection_data, grid, center, size,config)

        np.save('effmap_{}.npy'.format(ir), result)

    merge_effmap(0, rpet.nb_rings, rpet.nb_rings, 1, './')


if __name__ == '__main__':
    main()
