import numpy as np

class PSFKerenel():

    @classmethod
    def make_meshgrid(cls, grid, size):
        '''
        Create a 2D meshgrid locate at the origin of a Cartesian coordinate.

        Args:
            grid: the shape of the meshgrid.
            size: the region size of the meshgrid.
        Returns:
            xv: meshgrid in x axis.
            yv: meshgrid in y axis.
        '''
        sx, sy = size[0], size[1]
        nx, ny = grid[0], grid[1]
        ix, iy = sx/nx, sy/ny
        x = np.linspace((-sx+ix)/2, (sx-ix)/2, nx)
        y = np.linspace((-sy+iy)/2, (sy-iy)/2, ny)
        xv, yv = np.meshgrid(x,y, indexing = 'ij')
        return xv, yv

    @classmethod
    def make_polargrid(cls, xmesh, ymesh):
        '''
        Reprensent a 2D meshgrid in a polar coordinate.

        Args:
            xmesh: meshgrid in x axis.
            ymesh: meshgrid in y axis.
        
        Returns:
            rmesh: meshgrid in radial direction.
            pmesh: meshgrid in phi angle. 
        '''
        rmesh = np.zeros(xmesh.shape)
        pmesh = np.zeros(xmesh.shape)
        
        rmesh = np.sqrt(xmesh**2+ymesh**2)
        pmesh = np.arctan2(ymesh, xmesh)*180/np.pi
        return rmesh, pmesh


    @classmethod
    def compensate_kernel(cls, kernel_para, factor, xrange):
        '''
        Interpolate the kernel parameters in the whole range.
        
        Args:
            kernel_para: kerenl parameters to be interpolated.
            factor: scale ratio to be refined.
            xrange: the range of kernel.
        Returns:
            An interpolated kernel parameter array.
        '''
        from scipy import interpolate
        nb_samples = len(kernel_para)
        nb_new_samples = nb_samples*factor
        x = np.linspace(xrange[0], xrange[1], nb_samples)
        nb_columns = kernel_para.shape[1]
        kernel_new = np.zeros([nb_new_samples, nb_columns])
        for i_column in range(nb_columns):
            y = kernel_para[:, i_column]
            f = interpolate.interp1d(x, y)
            xnew = np.linspace(xrange[0], xrange[1], nb_new_samples)
            kernel_new[:,i_column] = f(xnew)
            
        return kernel_new

    @classmethod
    def locate_kernel(cls, sampled_kernels, xrange, r):
        '''
        Choose a approximated kernel for a mesh.
        
        Args:
            sampled_kernels: interpolated kernel parameter array
            xrange: the range of kerenl, xrange[0] is 0 generally.
            r: the radial distance to the origin
        
        Returns:
            A kernel image generated according to the samples.
        '''
        nb_samples = sampled_kernels.shape[2]
        interval = (xrange[1] - xrange[0])/(nb_samples-1)
        index = int(round((r-xrange[0])/interval))
        
        if index > nb_samples:
            print("index {} is out of sample domain.".format(index))
            kernel_image = np.zeros((sampled_kernels.shape[0], sampled_kernels.shape[1]))
        else:
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
    def preprocess_kernel_para(cls, origin_kernel, grid, size):
        '''
        Preprocess the kernel parameters.
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

        voxsize = size/grid
        kernel_XY = origin_kernel
        a = kernel_XY[:, 1]
        ux = (kernel_XY[:, 5]-round(grid[0]/2))*voxsize[0]
        sigmax = kernel_XY[:, 3]*voxsize[0]
        uy = (kernel_XY[:, 4]-round(grid[1]/2))*voxsize[1]
        sigmay = kernel_XY[:, 2]*voxsize[1]
        return np.vstack((a, ux, sigmax, uy, sigmay)).T
    
    @classmethod
    def compute_sample_kernels(cls, kernel_para, xmesh, ymesh):
        '''
        Compute the all the kernel images of sample points. 
        
        Args:
            kernel_para: kernel parameters to decide the distribution.
            xmesh: the meshgrid in x axis
            ymesh: the meshgrid in y axis
        
        Returns:
            
        '''
        nb_samples = len(kernel_para)
        grid = xmesh.shape
        kernel_images = np.zeros((grid[0], grid[1], nb_samples))
        
        for k in range(nb_samples):
            a = kernel_para[k,0]
            ux = kernel_para[k,1]
            sigx = kernel_para[k,2]
            uy = kernel_para[k,3]
            sigy = kernel_para[k,4]
            for i in range(grid[0]):
                for j in range(grid[1]):
                    kernel_images[i,j,k] = a*np.exp(-(xmesh[i,j]-ux)**2/(2*sigx**2)-(ymesh[i,j]-uy)**2/(2*sigy**2))
        return kernel_images

    @classmethod
    def compute_kernels(cls, kernel_para, grid, size, xrange, factor):
        """

        """
    #     print(kernel_para)    
        refined_kernel = cls.compensate_kernel(kernel_para, factor, xrange)
    #     print(refined_kernel.shape)

        xmesh, ymesh = cls.make_meshgrid(grid, size)
        xmesh = np.array(xmesh)
        ymesh = np.array(ymesh)
        
        kernel_sample_images = cls.compute_sample_kernels(refined_kernel, xmesh, ymesh)
        
        print(kernel_sample_images.shape)
        
    #     plt.figure()
    #     plt.imshow(kernel_sample_images[:,:, 90])
        return xmesh, ymesh, kernel_sample_images

if __name__ is '__main__':
    psf = PSFKerenel
    kernel_para = np.load('./maps_tor_1_4m/sigmaXY.npy')
    grid = np.array([195, 195, 416])
    size = np.array([666.9, 666.9, 1422.72])
    kernel_para = psf.preprocess_kernel_para(kernel_para, grid, size)
    grid = grid*1
    xrange = [0, 94*3.42]
    factor = 9
    xmesh, ymesh, kernel_samples = psf.compute_kernels(kernel_para, grid, size, xrange, factor)