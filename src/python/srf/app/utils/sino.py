from ...task.tasksino_infonew import SinoTaskSpec
import scipy.io as sio


def auto_osem_config(sino_task_config, distribute_config):
    """
    Auto complete lors shape and step.

    Arguments:
    - `tor_task_config`: dict of configs to init TorTaskSpec
    - `distribute_config`: dict of distribute_config
    """
    from dxl.learn.core import ClusterSpec
    from ...task.sinodatanew import SinoToRSpec,MatrixToRSpec
    import h5py
    ts = SinoTaskSpec(sino_task_config)
    cs = ClusterSpec(distribute_config)
    with h5py.File(ts.sino.path_file) as fin:
        sino = fin[ts.sino.path_dataset]
        shapes = sino.shape
        steps = shapes[0] //(cs.nb_workers ) 
        shapes = tuple([steps] + list(shapes[1:])) 
    new_sino_c = ts.sino.to_dict()
    new_sino_c['shapes'] = shapes
    new_sino_c['steps'] = steps
    ts.sino = SinoToRSpec(new_sino_c)


    fin = sio.loadmat(ts.matrix.path_file)
    matrix = fin[ts.matrix.path_dataset]
    # with h5py.File(ts.matrix.path_file) as fin:
    #     matrix = fin[ts.matrix.path_dataset]
    shapes = matrix.shape
    steps = shapes[0] //(cs.nb_workers ) 
    shapes = tuple([steps] + list(shapes[1:])) 
    new_matrix_c = ts.matrix.to_dict()
    new_matrix_c['shapes'] = shapes
    new_matrix_c['steps'] = steps
    ts.matrix = MatrixToRSpec(new_matrix_c)
    return ts.to_dict()
