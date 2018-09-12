import numpy as np
from dxl.learn.session import Session
from tqdm import tqdm

from srf.graph.reconstruction.ring_efficiency_map import RingEfficiencyMap
from srf.model import BackProjectionOrdinary
from srf.physics import CompleteLoRsModel
from srf.preprocess.merge_map import merge_effmap
from srf.function import make_scanner,create_listmode_data
from srf.data import ListModeDataWithoutTOF
import tensorflow as tf

def compute(lors, grid, center, size, kernel_width):
    physics_model = CompleteLoRsModel('map_model')
    backprojection_model = BackProjectionOrdinary(physical_model=physics_model)
    t = RingEfficiencyMap('effmap', backprojection_model, lors, grid, center, size)
    with Session() as sess:
        t.make()
        result = t.run()
    return result


def get_mct_config():
    """
    Retrun a dict that describes the geometry of mCT scanner.
    """
    config = {
        "modality": "PET",
        "name": "mCT",
        "ring": {
            "inner_radius": 424.5,
            "outer_radius": 444.5,
            "axial_length": 220.0,
            "nb_rings": 104,
            "nb_blocks_per_ring": 48,
            "gap": 0.0
        },
        "block": {
            "grid": [1, 13, 1],
            "size": [20.0, 53.3, 2.05],
            "interval": [0.0, 0.0, 0.0]
        },
        "tof": {
            "resolution": 530,
            "bin": 40
        }
    }
    return config


def main():
    grid = [128, 128, 110]
    center = [0.0, 0.0, 0.0]
    size = [262.4, 262.4, 225.5]
    kernel_width = 6.76
    config = get_mct_config()
    rpet = make_scanner('Cylinder',config)
    r1 = rpet.rings[0]

    for ir in tqdm(range(0, rpet.nb_rings)):
        print("start to compute the {} th map.".format(ir))
        r2 = rpet.rings[ir]
        lors = rpet.make_ring_pairs_lors(r1, r2)
        projection_data = create_listmode_data[ListModeDataWithoutTOF](lors)
        result = compute(projection_data, grid, center, size, kernel_width)
        np.save('effmap_{}.npy'.format(ir), result)

    merge_effmap(0, rpet.nb_rings, rpet.nb_rings, 1, './')





# def test_input_data_generate():
#     with tf.Graph().as_default():
#         grid = [128, 128, 110]
#         center = [0.0, 0.0, 0.0]
#         size = [262.4, 262.4, 225.5]
#         kernel_width = 6.76
#         rpet = make_scanner()
#         r1 = rpet.rings[0]
#         for ir in tqdm(range(0, min(rpet.nb_rings, 10))):
#             print("start to compute the {} th map.".format(ir))
#             r2 = rpet.rings[ir]
#             lors = rpet.make_ring_pairs_lors(r1, r2)
#             create_listmode_data[ListModeDataWithoutTOF](lors)


if __name__ == '__main__':
    main()
    # test_input_data_generate()
