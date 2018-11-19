import time

import numpy as np

from srf.scanner.pet.block import RingBlock


def merge_effmap(scanner, grid, center, size, z_factor, crop_ratio, path):
    """
    to do: implemented in GPU to reduce the calculated time
    """
    temp = np.load(path + 'effmap_{}.npy'.format(0)).T
    nb_image_layers = int(temp.shape[0])
    final_map = np.zeros(temp.shape)
    print(final_map.shape)
    st = time.time()
    nb_rings = scanner.nb_rings
    for ir in range(0, nb_rings):
        temp = np.load(path + 'effmap_{}.npy'.format(ir)).T
        print("process :{}/{}".format(ir + 1, nb_rings))
        for jr in range(nb_rings - ir):
            if ir == 0:
                final_map[jr:nb_image_layers, :, :] += temp[0:nb_image_layers - jr, :, :]
            else:
                final_map[jr:nb_image_layers, :, :] += temp[0:nb_image_layers - jr, :, :]
        et = time.time()
        tr = (et - st) / (nb_rings * (nb_rings - 1) / 2 - (nb_rings - ir - 1) * (
                nb_rings - ir - 2) / 2) * ((nb_rings - ir - 1) * (nb_rings - ir - 2) / 2)
        print("time used: {} seconds".format(et - st))
        print("estimated time remains: {} seconds".format(tr))

    # normalize the max value of the map to 1.
    # cut_start = int((nb_image_layers-nb_rings*z_factor)/2)
    # final_map = final_map[cut_start:nb_image_layers-cut_start,:,:]

    final_map = crop_image(scanner, final_map.T, grid, center, size, crop_ratio)

    final_map = final_map / np.max(final_map)
    final_map[final_map > 1e-7] = 1 / final_map[final_map > 1e-7]
    # final_map = final_map.T
    np.save(path + 'summap.npy', final_map)


def crop_image(scanner, image, grid, center, size, crop_ratio):
    """
    crop the effmap by the crop_ratio, the value of voxels out of the crop area are set to zero.

    Note: to main the consistency with the MLEM algorithm, the values are set to 1/value here and
          set the operation as multiple, that is, x/effmap->x*(1/effmap).
    """
    inner_radius = scanner.inner_radius
    half_height = scanner.axial_length / 2
    vox_meshes = make_meshes(size, grid, center)

    # print('the map shape is', vox_meshes.shape)
    xy_dis = np.sqrt(vox_meshes[:, 0] ** 2 + vox_meshes[:, 1] ** 2)
    z_dis = np.abs(vox_meshes[:, 2])
    image = image.reshape((-1, 1))

    image[np.where(xy_dis > crop_ratio * inner_radius)] = 0
    image[np.where(z_dis >= half_height)] = 0
    return image.reshape((grid[0], grid[1], grid[2]))


def merge_effmap_full(scanner, grid, center, size, z_factor, crop_ratio, path):
    """
    to do: implemented in GPU to reduce the calculated time
    """
    temp = np.load(path + './effmap/effmap_0_0.npy').T
    final_map = np.zeros(temp.shape)
    print(final_map.shape)
    st = time.time()
    nb_rings = scanner.nb_rings

    for ir1 in range(nb_rings):
        for ir2 in range(nb_rings):
            temp = np.load(path + f'./effmap/effmap_{ir1}_{ir2}.npy').T
            print(f"process :{ir1}/{nb_rings} and {ir2}/{nb_rings}")
            final_map += temp

    # normalize the max value of the map to 1.
    final_map = crop_image(scanner, final_map.T, grid, center, size, crop_ratio)
    final_map = final_map / np.max(final_map)
    final_map[final_map > 1e-7] = 1 / final_map[final_map > 1e-7]
    np.save(path + 'summap.npy', final_map)


def make_meshes(grid, center, size):
    """
    """
    block = RingBlock(grid, center, size, 0)
    return block.get_meshes()
