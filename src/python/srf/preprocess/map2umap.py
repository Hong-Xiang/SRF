import numpy as np
import math
import json

def _gray2material(filename):
    """
    from dat file get the relationship between grey value and material
    """
    f = open(filename,'r')
    line = f.readline()
    maping_relations = []
    while line:
        line = f.readline
        split_line = line.split(" ")
        maping_relation = _get_none_space_data(split_line)
        if maping_relation != []:
            maping_relations.append(maping_relation)
    return maping_relations

def _get_none_space_data(line):
    L = []
    k = 1
    for i in line:
        i = str.strip(i)
        if i:
            L.append(i)
            k = k+1
            if k>3:
                break
    return L

def _load_phantom(filename):
    return np.load(filename,dtype='float32')

def _get_range_lowlevel(relation):
    lowlevel = np.zeros((len(relation)))
    for i in range (0,len(relation)):
        lowlevel[i] = float(relation[i][0])
    return lowlevel

def cal_u_map(phantom_file,material_file,range_file):
    phantom = _load_phantom(phantom_file)
    u_map = np.zeros_like(phantom)
    relation = _gray2material(range_file)
    lowlevel = _get_range_lowlevel(relation)
    with open(material_file) as f:
        material2factor = json.load(f)
    for i in lowlevel:
        material = relation[i][2]
        tup = np.where(phantom>=i)
        u_map[tup] = material2factor[material]
    return u_map