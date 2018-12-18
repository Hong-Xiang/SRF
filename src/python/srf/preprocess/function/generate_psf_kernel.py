# this file aims to make the psf kernel of a cylindrical PET scanner.
import numpy as np
import math


class PSFMaker():
    '''
    A collection of the psf matrix creating process,
    Input the fitted xy and z kernel parameter(.h5 file with dict)
    Output a xy sparse .mat and a z dense .npy.
    '''

    def __init__(self):
        pass

    @classmethod
    def find_xy_kernel_max_sum(cls, kernel_samples):
        kernel_sum = np.sum(np.sum(kernel_samples, axis=0), axis=0)
        max_value = np.max(kernel_sum)
        print('max kernel xy sum value', max_value)
        return max_value

    @classmethod
    def find_z_kernel_max_sum(cls, kernel_samples):
        kernel_sum = np.sum(kernel_samples, axis=0)
        max_value = np.max(kernel_sum)
        print('max kernel z sum value', max_value)
        return max_value

    @classmethod
    def make_meshgrid(cls, grid, voxsize):
        nx, ny = grid[0], grid[1]
        ix, iy = voxsize[0], voxsize[1]
        sx, sy = ix*nx, iy*ny
        x = np.linspace((-sx+ix)/2, (sx-ix)/2, nx)
        y = np.linspace((-sy+iy)/2, (sy-iy)/2, ny)
        xv, yv = np.meshgrid(x, y, indexing='ij')
        return xv, yv

    @classmethod
    def make_polargrid(cls, xmesh, ymesh):
        rmesh = np.sqrt(xmesh**2+ymesh**2)
        pmesh = np.arctan2(ymesh, xmesh)*180/np.pi
        return rmesh, pmesh

    @classmethod
    def locate_kernel(cls, sampled_kernels, x_range, r):
        '''
        choose a approximated kernel for a mesh.

        Args:
            sampled_kernels: interpolated kernel parameter array
            x_max: the max x position of kernel samples.
            r_max: the radial distance of a pixel to the origin.

        Returns:
            A kernel image generated according to the samples.
        '''
        # kernel_image = sampled_kernels[:, :, int(
        #     round(r/(x_range/(sampled_kernels.shape[2]-1))))]
        # return kernel_image

        nb_samples = sampled_kernels.shape[2]
        interval = (x_range)/(nb_samples-1)
        index = int(round((r)/interval))
        
        if index >= nb_samples:
    #         print("index {} is out of sample domain.".format(index))
            kernel_image = np.zeros((sampled_kernels.shape[0], sampled_kernels.shape[1]))
        else:
    #         print("index:", index)
            kernel_image = sampled_kernels[:,:,index]
        return kernel_image

    @classmethod
    def rotate_kernel(cls, kernel_image, angle):
        '''
        rotate an 2D kernel in X-Y plane.
        Args:
            kernel_image: input kernel to be rotated
            angle: the rotation angle
        Returns:
            rotated kernel
        '''
        from scipy import ndimage
        img = kernel_image
        img_r = ndimage.interpolation.rotate(img, angle)
        shape0 = np.array(img.shape)
        shape1 = np.array(img_r.shape)
        # calculate the valid central part of kernel
        istart = np.round((shape1-shape0)/2)
        iend = shape0 + istart
        img_r = img_r[int(istart[0]):int(iend[0]), int(istart[1]):int(iend[1])]
        return img_r

    @classmethod
    def make_xy_kernel(cls, xmesh, ymesh, grid, x_range, kernel_samples, epsilon=1e-4):
        rmesh, pmesh = cls().make_polargrid(xmesh, ymesh)
        # import time
        nx, ny = grid[0], grid[1]
        row, col, data = [], [], []
        # start = time.time()
        kernel_max = cls().find_xy_kernel_max_sum(kernel_samples)
        from tqdm import tqdm
        for i in tqdm(range(nx)):
            for j in range(ny):
                irow = j + i*ny
                ir, ip = rmesh[i, j], pmesh[i, j]
                original_kernel = cls().locate_kernel(kernel_samples, x_range, ir)
                kernel = cls().rotate_kernel(original_kernel, ip)
                kernel = kernel/kernel_max
                kernel[kernel < epsilon] = 0.0
                kernel_flat = kernel.reshape((-1))
                from scipy import sparse
                spa_kernel = sparse.coo_matrix(kernel_flat)
                nb_ele = spa_kernel.data.size
                if nb_ele > 0:
                    row = np.append(row, [irow]*nb_ele)
                    col = np.append(col, spa_kernel.col)
                    data = np.append(data, spa_kernel.data)
        return np.array([row, col, data])

    @classmethod
    def compute_sample_kernels(cls, kernel_array, xmesh, ymesh):
        '''
        Compute the all the kernel images of sample points. 

        Args:
            kernel_para: kernel parameters to decide the distribution.
            xmesh: the meshgrid in x axis
            ymesh: the meshgrid in y axis

        Returns:

        '''
        nb_samples = len(kernel_array)
        grid = xmesh.shape
        kernel_images = np.zeros((grid[0], grid[1], nb_samples))
        from tqdm import tqdm
        for k in tqdm(range(nb_samples)):
            a = kernel_array[k, 0]
            ux = kernel_array[k, 1]
            sigx = kernel_array[k, 2]
            uy = kernel_array[k, 3]
            sigy = kernel_array[k, 4]
            for i in range(grid[0]):
                for j in range(grid[1]):
                    kernel_images[i, j, k] = a*np.exp(-(xmesh[i, j]-ux)**2/(
                        2*sigx**2)-(ymesh[i, j]-uy)**2/(2*sigy**2))
        return kernel_images

    @classmethod
    def compensate_kernel(cls, kernel_array, factor, xrange):
        '''
        The experimental xy_kernel parameters were corase along the x axis.
        Interpolate the kernel parameters in the whole range.

        Args:
            kernel_array: kerenl parameters to be interpolated.
            factor: scale ratio to be refined.
            xrange: the x axis range of kernel.
        Returns:
            An interpolated kernel parameter array.
        '''
        from scipy import interpolate
        nb_samples = len(kernel_array)
        nb_new_samples = int(nb_samples*factor)
        x = np.linspace(0, xrange, nb_samples)
        nb_columns = kernel_array.shape[1]

        kernel_new = np.zeros([nb_new_samples, nb_columns])
        for i_column in range(nb_columns):
            y = kernel_array[:, i_column]
            f = interpolate.interp1d(x, y)
            xnew = np.linspace(0, xrange, nb_new_samples)
            kernel_new[:, i_column] = f(xnew)
        return kernel_new

    @classmethod
    def preprocess_xy_para(cls, kernel_para_dict, grid, voxsize):
        '''
        Preprocess the kernel parameters.
        The kernel is shifted to the standard coordinate.
        The amplitude is changed in this process, but it 
        does not matter since the kernel will be normalized
        later.

        Args:
            origin_kernel: the origin kernel fitted from the experiment.
            grid: kernel image grid
            voxsize: voxel size
        Returns:
            processed kernel parameter array.

        '''
        origin_kernel = kernel_para_dict
        if not isinstance(origin_kernel, dict):
            raise TypeError('The kernel parameter should be dictionary.')
        else:
            axy = origin_kernel['axy']
            ux = origin_kernel['ux']
            uy = origin_kernel['uy']
            sigmax = origin_kernel['sigmax']
            sigmay = origin_kernel['sigmay']
            # kernel_XY = origin_kernel
            # axy = kernel_XY[:, 1]
            ux = (ux-math.ceil(grid[0]/2))*voxsize[0]
            sigmax = sigmax*voxsize[0]
            uy = (uy-math.ceil(grid[1]/2))*voxsize[1]
            sigmay = sigmay*voxsize[1]
            return np.vstack((axy, ux, sigmax, uy, sigmay)).T
        # raise DeprecationWarning("this function is going to be deprecated")

    @classmethod
    def xy_main(cls, kernel_para_dict: dict,  grid: list, voxsize: list, refined_factor: int, output_mat_dir):
        # step1: generate kernel array
        # print(type(kernel_para_dict))
        kernel_array = cls().preprocess_xy_para(kernel_para_dict, grid, voxsize)
        x_range = voxsize[0]*(kernel_array.shape[0]-1)
        # step2: interpolation
        refined_kernel = cls().compensate_kernel(kernel_array, refined_factor, x_range)
        xmesh, ymesh = cls().make_meshgrid(grid, voxsize)
        # step3: compute whole kernel
        kernel_samples = cls().compute_sample_kernels(refined_kernel, xmesh, ymesh)
        print(kernel_samples.shape)
        # np.save(output_kernel_dir, kernel_samples)
        # kernel_samples = np.load(output_kernel_dir)

        kernel = cls().make_xy_kernel(xmesh, ymesh, grid,
                                      x_range, kernel_samples, epsilon=1e-4)
        row, col, data = kernel[0], kernel[1], kernel[2]
        row = row.astype(int)
        col = col.astype(int)
        from scipy import sparse
        kernel_xy = sparse.coo_matrix((data, (row, col)), shape=(
            grid[0]*grid[1], grid[0]*grid[1]), dtype=np.float32)
        # print(kernel_xy.size)

        import scipy.io as sio
        sio.savemat(output_mat_dir, {'matrix': kernel_xy})
        print(f'siddon kernel_xy size = {kernel_xy.size}')

    @classmethod
    def compute_z_sample_kernels(cls, kernel_samples, grid_z, zmesh):
        nb_samples = len(kernel_samples)
        kernel_z = np.zeros((grid_z, grid_z), dtype=np.float32)
        for k in range(nb_samples):
            az = kernel_samples[k, 0]
            uz = kernel_samples[k, 1]
            sigz = kernel_samples[k, 2]
            for i in range(grid_z):
                kernel_z[i, k] = az*np.exp(-(zmesh[i]-uz)**2/(2*sigz**2))
        
        kernel_max =cls().find_z_kernel_max_sum(kernel_z)
        return kernel_z/kernel_max

    @classmethod
    def preprocess_z_para(cls, original_para, grid_z, voxsize_z):
        '''
        Preprocess the kernel parameters in z.
        The kernel is shifted to the standard coordinate.
        The amplitude is changed in this process, but it 
        does not matter since the kernel will be normalized
        later.

        Args:
            origin_kernel: the origin kernel fitted from the 
            experiment.
            grid: kernel image grid
            size: image domain size
        Returns:
            processed kernel parameter array.

        '''
        if not isinstance(original_para, dict):
            raise TypeError('The kernel parameter should be dictionary.')
        else:
            az = np.array(original_para['az'])
            uz = np.array((original_para['uz']-math.ceil(grid_z/2))*voxsize_z)
            sigmaz = np.array(original_para['sigmaz']*voxsize_z)

            az_neg = np.flipud(az[1:])
            uz_neg = -np.flipud(uz[1:])
            sigmaz_neg = np.flipud(sigmaz[1:])

            compen_len = int((grid_z-len(az)*2+1)/2)
            # print(compen_len)
            az = np.hstack(
                ([0.0] * compen_len, az_neg, az, [0.0] * compen_len))
            uz = np.hstack(
                ([0.0] * compen_len, uz_neg, uz, [0.0] * compen_len))
            sigmaz = np.hstack(
                ([1.0] * compen_len, sigmaz_neg, sigmaz, [1.0] * compen_len))

            return np.vstack((az, uz, sigmaz)).T

    @classmethod
    def z_main(cls, kernel_para_dict, grid_z, voxsize_z, output_kernel_dir):
        zmesh = np.linspace((-voxsize_z*(grid_z-1))/2,
                            (voxsize_z*(grid_z-1))/2, grid_z)
        kernel_samples = cls().preprocess_z_para(kernel_para_dict, grid_z, voxsize_z)
        kernel_z = cls().compute_z_sample_kernels(kernel_samples, grid_z, zmesh)
        np.save(output_kernel_dir, kernel_z)
        # return kernel_z

    @classmethod
    def run(cls, kernel_xy_para_dir, kernel_z_para_dir,
            grid: list, voxsize: list, refined_factor,
            output_mat_dir: str, output_kernel_dir: str,
            map_file:str, psf_map_file:str, epsilon:float):
        from srf.io.listmode import load_h5
        grid = np.array(grid)
        voxsize = np.array(voxsize) 
        kernel_xy_dict = load_h5(kernel_xy_para_dir)
        kernel_z_dict = load_h5(kernel_z_para_dir)
        print("step 1/3: making xy kernel...")
        cls().xy_main(kernel_xy_dict,
                      grid[0:2], voxsize[0:2], refined_factor, output_mat_dir)
        
        print("step 2/3: making z kernel...")
        cls().z_main(kernel_z_dict, grid[2], voxsize[2], output_kernel_dir)
        print("step 3/3: generating psf efficiency map...")
        cls().map_process(output_mat_dir, output_kernel_dir, map_file, psf_map_file, epsilon)
        print("Task complete!")

    @classmethod
    def map_process(cls, xy_mat_dir, z_dense_dir, map_file, psf_map_file, epsilon):
        import scipy.io as sio
        from scipy import sparse
        kernel_xy = sparse.coo_matrix(sio.loadmat(xy_mat_dir)['matrix'])
        # print(kernel_xy)
        kernel_z = np.load(z_dense_dir)
        # plt.figure(figsize = (16,16))
        # plt.imshow(kernel_z)
        # plt.colorbar()
        effmap = np.load(map_file)
        # print('max effmap value', np.max(effmap))
        # print('min effmap value', np.min(effmap))
        # mask = effmap >1e5
        # effmap[effmap > 700] = 0

        effmap[effmap>=1] = 1/effmap[effmap>=1]
        # print('max effmap value', np.max(effmap))
        # print('min effmap value', np.min(effmap))
        # effmap[effmap <= 1e-7] = 0
        grid = effmap.shape
        effmap_reshaped = effmap.reshape((-1, grid[2]))

        z_effmap = np.matmul(effmap_reshaped, kernel_z)
        # kernel_xy transpose
        kernel_xy = sparse.coo_matrix((kernel_xy.data,(kernel_xy.col, kernel_xy.row)), 
                                    shape = (grid[0]*grid[1], grid[0]*grid[1]), dtype = np.float32)

        # print(kernel_xy)
        psf_effmap = kernel_xy.dot(z_effmap)

        # z_effmap = z_effmap.reshape(grid)
        # z_effmap = z_effmap/np.max(z_effmap)
        # print('max psfz effmap value', np.max(z_effmap))
        # print('min psfz effmap value', np.min(z_effmap))
        # z_effmap[z_effmap <= 1e-7] = 0
        # z_effmap[z_effmap>1e-7] = 1/z_effmap[z_effmap>1e-7]
        # print('max psfz effmap value', np.max(z_effmap))
        # print('min psfz effmap value', np.min(z_effmap))


        psf_effmap = psf_effmap.reshape(grid)

        psf_effmap = psf_effmap/np.max(psf_effmap)
        # print('max psf effmap value', np.max(psf_effmap))
        # print('min psf effmap value', np.min(psf_effmap))
        psf_effmap[psf_effmap <= epsilon] = 0
        psf_effmap[psf_effmap>epsilon] = 1/psf_effmap[psf_effmap>epsilon]
        print('max psf effmap value', np.max(psf_effmap))
        print('min psf effmap value', np.min(psf_effmap))

        # effmap[effmap> 1e-7] = 1/effmap[effmap> 1e-7]

        # np.save('exp_short_siddon_map_psfz.npy', z_effmap)
        np.save(psf_map_file, psf_effmap)
