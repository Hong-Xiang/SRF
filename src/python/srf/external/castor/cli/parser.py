import json
from srf.external.castor.io import load_config
__all__ = ["parse_recon", "parse_root_to_castor", "parse_mac_to_geom"]

def _append_str(cmd: str, abbr_str, option_term):
    if option_term is None:
        cmd = cmd + ' '
    else:
        cmd = cmd + ' ' + abbr_str + ' ' + option_term
    return cmd

# def _parser(config:dict, cmd:str):
#     for item in config.items():
#         if not isinstance(item[1], dict) and not isinstance(item[1], list):
#             _append_str(cmd, )


def _append_list(vec_list: list):
    vec_str = str(vec_list[0])
    for item in vec_list[1:]:
        vec_str += ','+str(item)
    return vec_str


def parse_recon(config_file):
    config = load_config(config_file)
    execute = 'castor-recon'
    cmd = execute

    cmd = _append_str(cmd, '-df', config['datafile'])
    cmd = _append_str(cmd, '-opti', config['optimization'])

    i = config['iteration']
    iter_str = str(i['nb_iter']) + ':' + str(i['subsets'])
    cmd = _append_str(cmd, '-it', iter_str)
    cmd = _append_str(cmd, '-proj', config['projector'])

    if config['convolution'] is not None:
        c = config['convolution']
        conv_str = c['filter'] + ',' + _append_list(c['size'])+'::psf'
        cmd = _append_str(cmd, '-conv', conv_str)

    cmd = _append_str(cmd, '-dim', _append_list(config['dimension']))
    cmd = _append_str(cmd, '-vox', _append_list(config['voxel_size']))
    cmd = _append_str(cmd, '-dout', config['data_out'])
    return cmd

def parse_root_to_castor(config_file):
    config = load_config(config_file)
    execute = 'castor-GATERootToCastor'
    cmd = execute
    c = config['input']
    if c['list'] is not None:
        cmd = _append_str(cmd, '-il', c['list'])
    elif c['file'] is not None:
        cmd = _append_str(cmd, '-i', c['file'])
    else:
        raise ValueError('No input root data file(s) for this convertion!')
    cmd = _append_str(cmd, '-o', config['output'])
    cmd = _append_str(cmd, '-m', config['macro_file'])
    cmd = _append_str(cmd, '-s', config['scanner_alias'])

    return cmd

def parse_listmode_to_castor(config_file):
    return load_config(config_file)


def parse_sino_to_castor(config_file):
    config = load_config(config_file)
    execute = 'cgy-GATESinotoCastor'
    cmd = execute
    c = config['input']
    if c['list'] is not None:
        cmd = _append_str(cmd, '-il', c['list'])
    elif c['file'] is not None:
        cmd = _append_str(cmd, '-i', c['file'])
    else:
        raise ValueError('No input root data file(s) for this convertion!')
    cmd = _append_str(cmd, '-o', config['output'])
    cmd = _append_str(cmd, '-m', config['macro_file'])
    cmd = _append_str(cmd, '-s', config['scanner_alias'])
    cmd = _append_str(cmd, '-t', config['true_flag'])

    return cmd


def parse_mac_to_geom(config_file):
    config = load_config(config_file)
    execute = 'castor-GATEMacToGeom'
    cmd = execute

    cmd = _append_str(cmd, '-m', config['macro_file'])
    cmd = _append_str(cmd, '-o', config['output'])
    
    return cmd

