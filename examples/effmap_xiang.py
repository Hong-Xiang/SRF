import numpy as np
from dxl.learn.session import Session
from tqdm import tqdm

from srf.data import ListModeDataWithoutTOF
from srf.function import create_listmode_data
from srf.graph.reconstruction.ring_efficiency_map import RingEfficiencyMap
from srf.model import BackProjectionOrdinary
from srf.physics import CompleteLoRsModel
from srf.preprocess.merge_map import merge_effmap
from srf.scanner.pet.block import Block
from srf.scanner.pet.geometry import RingGeometry
from srf.scanner.pet.pet import CylindricalPET
from srf.scanner.pet.spec import TOF


def compute(lors, grid, center, size):
    # physics_model = SplitLorsModel('map_model', kernel_width=kernel_width)
    physics_model = CompleteLoRsModel('map_model')
    backprojection_model = BackProjectionOrdinary(physical_model=physics_model)
    # map_step = MapStep('compute/effmap', backprojection=backprojection_model)

    # t = RingEfficiencyMap('effmap', compute_graph=map_step,
    #                       lors=lors, grid=grid,
    #                       center=center, size=size)
    t = RingEfficiencyMap('effmap', backprojection_model, lors, grid, center, size)
    with Session() as sess:
        t.make()
        result = t.run()
    # print(result[result>0])
    # sess.reset()
    return result


def get_mct_config():
    """
    Retrun a dict that describes the geometry of mCT scanner.
    """
    config = {
        "modality": "PET",
        "name": "mCT",
        "ring": {
            "inner_radius": 99.0,
            "outer_radius": 119.0,
            "axial_length": 33.4,
            "nb_rings": 10,
            "nb_blocks_per_ring": 16,
            "gap": 0.0
        },
        "block": {
            "grid": [1, 10, 10],
            "size": [20.0, 33.4, 33.4],
            "interval": [0.0, 0.0, 0.0]
        },
        "tof": {
            "resolution": 500000,
            "bin": 40
        }
    }
    return config


def make_scanner():
    """
    Create a cylindrical PET scanner.
    """
    config = get_mct_config()
    ring = RingGeometry(config['ring'])
    block = Block(block_size=config['block']['size'],
                  grid=config['block']['grid'])
    name = config['name']
    tof = TOF(res=config['tof']['resolution'], bin=config['tof']['bin'])
    return CylindricalPET(name, ring, block, tof)


def main():
    center = [0.0, 0.0, 0.0]
    size = [220.0, 220.0, 33.4]
    grid = [110, 110, 20]
    rpet = make_scanner()
    r1 = rpet.rings[0]

    for ir in tqdm(range(0, rpet.nb_rings)):
        print("start to compute the {} th map.".format(ir))
        r2 = rpet.rings[ir]
        lors = rpet.make_ring_pairs_lors(r1, r2)
        projection_data = create_listmode_data[ListModeDataWithoutTOF](lors)
        result = compute(projection_data, grid, center, size)

        np.save('effmap_{}.npy'.format(ir), result)

    merge_effmap(0, rpet.nb_rings, rpet.nb_rings, 1, './')


if __name__ == '__main__':
    main()