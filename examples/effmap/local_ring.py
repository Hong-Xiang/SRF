import time
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from srf.scanner.pet.pet import CylindricalPET
from srf.scanner.pet.geometry import RingGeometry
from srf.scanner.pet.spec import TOF
from srf.scanner.pet.block import Block,RingBlock
from srf.preprocess.function.on_tor_lors import map_process

from srf.graph.reconstruction.ring_efficiency_map import RingEfficiencyMap

from srf.model.backprojection import BackProjectionToR
from srf.model.map_step import MapStep
from srf.physics import ToRMapModel
from srf.preprocess.merge_map import merge_effmap
from dxl.learn.core import Session

def compute(lors, grid, center, size, kernel_width):

    physics_model = ToRMapModel('map_model', kernel_width=kernel_width)
    backprojection_model = BackProjectionToR(projection_model=physics_model)
    map_step = MapStep('compute/effmap', backprojection=backprojection_model)

    t = RingEfficiencyMap('effmap', compute_graph=map_step,
                          lors=lors, grid=grid,
                          center=center, size=size)
    t.make()
    with Session() as sess:
        result = t.run()
        print(result[result>0])
    sess.reset()
    return result


def get_mct_config():
    """
    Retrun a dict that describes the geometry of mCT scanner.
    """
    config = {
            "modality":"PET",
            "name": "mCT",
            "ring":{
                "inner_radius": 424.5,
                "outer_radius": 444.5,
                "axial_length": 220.0,
                "nb_rings": 104,
                "nb_blocks_per_ring": 48,
                "gap": 0.0
            },
            "block":{
                "grid":[1, 13, 1],
                "size":[20.0, 53.3, 2.05],
                "interval": [0.0, 0.0, 0.0]
            },
            "tof":{
                "resolution":530,
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
    # modality = config['modality']
    tof = TOF(res=config['tof']['resolution'], bin=config['tof']['bin'])
    return CylindricalPET(name, ring, block, tof)


def main():
    grid = [128, 128, 104]
    center = [0.0, 0.0, 0.0]
    size = [262.4, 262.4, 213.2]
    kernel_width = 7.76
    rpet = make_scanner()
    r1 = rpet.rings[0]

    for ir in tqdm(range(0, rpet.nb_rings)):
        print("start to compute the {} th map.".format(ir))
        r2 = rpet.rings[ir]
        lors = rpet.make_ring_pairs_lors(r1, r2)
        # np.save('debug_lors.npy',lors)
        lors = map_process(lors)

        from srf.preprocess.function.on_tor_lors import Axis as AXIS
        # print("preprocessed lors number is:",lors[AXIS.x].shape[0]+lors[AXIS.y].shape[0]+lors[AXIS.z].shape[0])
        # np.save("processed_lors.npy", lors[AXIS.x])
        # print(lors[AXIS.y][0,:])
        # print(lors[AXIS.z][0,:])
        result = compute(lors, grid, center, size, kernel_width)

        np.save('effmap_{}.npy'.format(ir), result)
    
    merge_effmap(0, rpet.nb_rings, rpet.nb_rings, 1, './')


if __name__ == '__main__':
    main()
