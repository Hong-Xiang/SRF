from ...task.task_info import ToRTaskSpec


def auto_osem_config(tor_task_config, distribute_config):
    """
    Auto complete lors shape and step.

    Arguments:
    - `tor_task_config`: dict of configs to init TorTaskSpec
    - `distribute_config`: dict of distribute_config
    """
    from dxl.learn.core import ClusterSpec
    from ...task.data import LoRsToRSpec
    import h5py
    ts = ToRTaskSpec(tor_task_config)
    cs = ClusterSpec(distribute_config)
    lors_c = ts.lors.to_dict()
    with h5py.File(ts.lors.path_file) as fin:
        lors3 = fin[ts.lors.path_dataset]
        XYZ = ['x', 'y', 'z']
        shapes = {a: lors3[a].shape for a in XYZ}
        steps = {a: shapes[a][0] //
                 (cs.nb_workers * ts.osem.nb_subsets) for a in XYZ}
        shapes = {a: tuple([steps[a]] + list(shapes[a][1:])) for a in XYZ}
    new_lors_c = ts.lors.to_dict()
    new_lors_c['shapes'] = shapes
    new_lors_c['steps'] = steps
    ts.lors = LoRsToRSpec(new_lors_c)
    return ts.to_dict()
