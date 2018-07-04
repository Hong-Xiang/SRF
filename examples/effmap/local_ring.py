import time
import tensorflow as tf 
from tqdm import tqdm

from srf.scanner.pet.pet import CylindricalPET
from srf.scanner.pet.geometry import RingGeometry
from srf.scanner.pet.spec import TOF 
from srf.scanner.pet.block import Block
from srf.preprocess import preprocess

from srf.graph.reconstruction.ring_effciency_map import RingEfficiencyMap

from srf.model.backprojection import BackProjectioTOR
from srf.model.map_step import MapStep 
from srf.physics.tor_map import ToRMapModel


def compute():

    model = ToR    

    config = tf.ConfigProto
    config.gpu_option.allow_growth = True
    t = RingEfficiencyMap()
    result = t.tensors[t.KT.TENSOR.RESULT]
    t.make()
    with tf.Session(config=config) as sess:
        result = sess.run(result)
    tf.reset_default_graph()
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
                "nb_rings": 4,
                "nb_blocks_per_ring": 48,
                "gap": 4.0
        },
        "block": {
            "grid": [1, 13, 13],
            "size": [20.0, 52.0, 52.0],
            "interval": [0.0, 0.0, 0.0]
        },
        "tof": {
            "resolution": 530,
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
    block = Block(block_size = config['block']['size'],
                    grid = config['block']['grid'])
    name = config['name']
    # modality = config['modality']
    tof = TOF(res = config['tof']['resolution'], bin = config['tof']['bin'])
    return CylindricalPET(name, ring, block, tof)    



def main():
    kernel_width = 7.76
    grid = [90, 90, 24]
    center = [0.0, 0.0, 0.0]
    size = [150.3, 150.3, 40.08]
    rpet = make_scanner()

    r1 = rpet.rings[0]
    
    for ir in tqdm(range(0,  rpet.nb_rings)):
        print("start to compute the {} th map.".format(ir))
        r2 = rpet.rings[ir]
        lors = rpet.make_ring_pairs_lors(r1, r2)
        lors = preprocess(lors)
        
        effmap = compute(lors, )
        np.save('effmap_{}.npy'.format(ir), result)

        







    