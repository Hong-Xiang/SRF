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

def _load_image(filename):
    return np.load(filename,dtype='float32')

def _get_range_lowlevel(relation):
    lowlevel = np.zeros((len(relation)))
    for i in range (0,len(relation)):
        lowlevel[i] = float(relation[i][0])
    return lowlevel

def cal_umap_from_simu(phantom_file,material_file,range_file):
    phantom = _load_image(phantom_file)
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

def cal_umap_from_ct(ct_image,voltage):
    image = _load_image(ct_image)
    u_map = np.zeros_like(phantom)
    index = np.where(image<0)
    u_map[index[0],index[1],index[2]] = (image[index[0],index[1],index[2]]/1000+1)*0.0096
    index = np.where(image>=0)
    coefficient = _get_coeff_from_volt(voltage)
    u_map[index[0],index[1],index[2]] = (image[index[0],index[1],index[2]]*coefficient+1)*0.0096
    return u_map

def _get_coeff_from_volt(voltage):
    if voltage == 140:
        return 6.4*10**-4
    elif voltage == 130:
        return 6.05*10**-4
    elif voltage == 120:
        return 5.76*10**-4
    elif voltage == 100:
        return 5.09*10**-4