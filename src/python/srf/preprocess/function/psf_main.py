# encoding: utf-8
'''
@author: Minghao Guo, Xiang Hong, Gaoyu Chen and Weijie Tao
@license: LGPL_v3.0
@contact: mh.guo0111@gmail.com
@software: srf_v2
@file: psf_main.py
@date: 12/17/2018
@desc: new version of Scalable Reconstruction Framework for Medical Imaging
'''
import json
import numpy as np
from srf.psf.gaussian_fit import fit_gaussian
from tqdm import tqdm
import click
import h5py


@click.command()
@click.option('--config', '-c', type = click.Path(exists = True))
@click.option('--mode', '-m', type = str, default = 'fit_mu')
@click.option('--half_slice_range', type = int, default = 2)
@click.option('--half_patch_range', type = int, default = 4)
@click.option('--out_dir', '-o', type = str, default = '')
def main(config, mode, half_slice_range, half_patch_range, out_dir):
    with open(config, 'r') as fin:
        print(config)
        config = json.load(fin)
        psf_xy = config['psf']['psf_xy']
        psf_z = config['psf']['psf_z']
        if not out_dir:
            out_path = config['psf']['out_dir']
        else:
            out_path = out_dir
        # psf xy fitting
        n_xy = len(psf_xy['pos'])
        out_xy = np.zeros((n_xy, 5))
        for i in tqdm(range(n_xy)):
            path = psf_xy['path_prefix'] + str(psf_xy['ind'][i]) + psf_xy['path_postfix']
            img = np.load(path)
            pos = psf_xy['pos'][i]
            nx, ny, nz = img.shape
            img_average = np.average(
                img[pos[0] - half_patch_range: min(pos[0] + half_patch_range + 1, nx),
                pos[1] - half_patch_range: min(pos[1] + half_patch_range + 1, ny),
                pos[2] - half_slice_range: min(pos[2] + half_slice_range + 1, nz)], axis = 2)
            # y, x = np.meshgrid(np.arange(img_average.shape[1]), np.arange(img_average.shape[0]))
            y, x = np.meshgrid(
                np.arange(pos[1] - half_patch_range, min(pos[1] + half_patch_range + 1, ny)),
                np.arange(pos[0] - half_patch_range, min(pos[0] + half_patch_range + 1, nx)))
            out_xy[i, :] = fit_gaussian(img_average, (x, y), mode = mode, mu = pos[:2])
        out_xy[:, 1:3] = np.sqrt(out_xy[:, 1:3])

        # psf z fitting
        n_z = len(psf_z['pos'])
        out_z = np.zeros((n_z, 3))
        for i in tqdm(range(n_z)):
            path = psf_z['path_prefix'] + str(psf_z['ind'][i]) + psf_z['path_postfix']
            img = np.load(path)
            pos = psf_z['pos'][i]
            nx, ny, nz = img.shape
            img_average = np.average(
                img[pos[0] - half_slice_range: min(pos[0] + half_slice_range + 1, nx),
                pos[1] - half_slice_range: min(pos[1] + half_slice_range + 1, ny),
                pos[2] - half_patch_range: min(pos[2] + half_patch_range + 1, nz)], axis = (0, 1))
            z = np.arange(pos[2] - half_patch_range, min(pos[2] + half_patch_range + 1, nz))
            out_z[i, :] = fit_gaussian(img_average, z, mode = 'fit_mu', mu = pos[2])
        out_z[:, 1] = np.sqrt(out_z[:, 1])

        with h5py.File(out_path, 'w') as fout:
            fout.create_dataset('out_xy', data = out_xy)
            fout.create_dataset('out_z', data = out_z)


if __name__ == "__main__":
    main()
