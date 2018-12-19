import json

with open('API_siddon.json', 'r') as fin:
    config = json.load(fin)
fitting = {'psf_xy': {}, 'psf_z': {}}
fitting['psf_xy']['out_dir'] = 'fit_para_xy.npy'
fitting['psf_xy']['path_prefix'] = '/mnt/gluster/Techpi/long/data_1.4_xy/recon_14_'
fitting['psf_xy']['path_postfix'] = '.npy'
fitting['psf_xy']['mode'] = 'fit_mu'
fitting['psf_xy']['half_slice_range'] = 2
fitting['psf_xy']['half_patch_range'] = 4

pos = []
ind = []
for i in range(100):
    pos.append([i + 97, 97, 212])
    ind.append(i)
fitting['psf_xy']['pos'] = pos
fitting['psf_xy']['ind'] = ind

fitting['psf_z']['out_dir'] = 'fit_para_z.npy'
fitting['psf_z']['path_prefix'] = '/mnt/gluster/Techpi/long/data_1.4_z/recon_14_'
fitting['psf_z']['path_postfix'] = '.npy'
fitting['psf_z']['mode'] = 'fit_mu'
fitting['psf_z']['half_slice_range'] = 2
fitting['psf_z']['half_patch_range'] = 4

pos = []
ind = []
for i in range(201):
    pos.append([97, 97, 212 + i])
    ind.append(i)
fitting['psf_z']['pos'] = pos
fitting['psf_z']['ind'] = ind

kernel = {}
kernel['grid'] = [195, 195, 425]
kernel['voxel_size'] = [3.42, 3.42, 3.42]
kernel['fit_para_xy_path'] = fitting['psf_xy']['out_dir']
kernel['fit_para_z_path'] = fitting['psf_z']['out_dir']
kernel['x_refined_factor'] = 7
kernel['kernel_xy_path'] = './kernel_xy.mat'
kernel['kernel_z_path'] = './kernel_z_dense.npy'

map = {}
map['kernel_xy_path'] = kernel['kernel_xy_path']
map['kernel_z_path'] = kernel['kernel_z_path']
map['map_path'] = './exp_short_siddon_map.npy'
map['psf_map_path'] = 'exp_short_siddon_map_psf.npy'
map['epsilon'] = 1e-7

config['psf'] = {'fitting': fitting, 'kernel': kernel, 'map': map}

with open('API_psf.json', 'w') as fout:
    json.dump(config, fout, indent = 4, separators = (',', ': '))
