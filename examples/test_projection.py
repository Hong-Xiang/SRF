import numpy as np
from dxl.core.debug import enter_debug
from dxl.learn.session import Session
from dxl.learn.tensor import const
from srf.data import Image, ListModeData
from srf.io.listmode import load_h5, save_h5
from srf.model import ProjectionOrdinary
from srf.physics import ProjectionLoRsModel
import tensorflow as tf
import json
enter_debug()

RESOURCE_ROOT = '/mnt/gluster/Techpi/brain16/recon/data/'


def main(task_config):
    with Session() as sess:
        al_config = task_config['algorithm']['projection_model']
        model = ProjectionLoRsModel('model', **al_config['siddon'])
        image0 = Image(const[tf](np.load('./image0.npy').astype(np.float32)), center = [0,0,0], size=[220,220,30])
        data = load_h5(task_config['input']['listmode']['path_file'])
        lors_point = np.hstack((data['fst'], data['snd']))
        lors = np.hstack((lors_point, data['weight'].reshape(data['weight'].size, 1)))

        projection_data = ListModeData(lors, np.ones([lors.shape[0]], np.float32))
        proj = ProjectionOrdinary(model)(image0, projection_data)
        result = sess.run(proj.values)

        # np.save('./proj.npy', np.hstack((lors_point, result)))
        data.update({"weight": result})
        save_h5("d18new.h5", data)




if __name__ == "__main__":
    with open("./API0.json") as fin:
        config = json.load(fin)
    main(config)
