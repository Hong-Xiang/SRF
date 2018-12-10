# this file aims to make the psf kernel of a cylindrical PET scanner.
import numpy as np
class PSFMaker():
    '''
    a 
    '''
    def __init__(self, xy_kernel_para:dict, z_kernel_para:dict):
        self.xy_kernel_para = xy_kernel_para
        self.z_kernel_para = z_kernel_para
    
    # @classmethod
    def make_meshgrid(self, grid, size):
        sx, sy = size[0], size[1]
        nx, ny = grid[0], grid[1]
        ix, iy = sx/nx, sy/ny
        x = np.linspace((-sx+ix)/2, (sx-ix)/2, nx)
        y = np.linspace((-sy+iy)/2, (sy-iy)/2, ny)
        xv, yv = np.meshgrid(x,y, indexing = 'ij')
    
    # @classmethod
    def make_polargrid(self, xmesh, ymesh):
        rmesh = np.sqrt(xmesh**2+ymesh**2)
        pmesh = np.arctan2(ymesh, xmesh)*180/np.pi
        return rmesh, pmesh
    
    def compensate_kernel(self, kernel_para, factor, xrange):
        '''
        The experimental xy_kernel parameters were corase along the x axis.
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
            y = kernel_para[:,i_column]
            f = interpolate.interp1d(x,y)
            xnew = np.linspace(xrange[0], xrange[1], nb_new_samples)
            kernel_new[:,i_column] = f(xnew)
        
        return kernel_new
    
    def locate_kernel(self, sampled_kernels, xrange, r):
        '''
        choose a approximated kernel for a mesh.
        
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

        if index >= nb_samples:
    #         print("index {} is out of sample domain.".format(index))
            kernel_image = np.zeros((sampled_kernels.shape[0], sampled_kernels.shape[1]))
        else:
    #         print("index:", index)
            kernel_image = sampled_kernels[:,:,index]
        return kernel_image  

    def rotate_kernel(self, kernel_image, angle):
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
    
    #TODO: this function is going to be deprecated 
    def preprocess_kernel_para(self, origin_kernel, grid, size):
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
        ux = (kernel_XY[:, 5]-math.ceil(grid[0]/2))*voxsize[0]
        sigmax = kernel_XY[:, 3]*voxsize[0]
        uy = (kernel_XY[:, 4]-math.ceil(grid[1]/2))*voxsize[1]
        sigmay = kernel_XY[:, 2]*voxsize[1]
        return np.vstack((a, ux, sigmax, uy, sigmay)).T
        # raise DeprecationWarning("this function is going to be deprecated")
    
    def compute_sample_kernels(self, kernel_para, xmesh, ymesh):
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
    
    def compute_kernels(self, kernel_para, grid, size, xrange, factor):  
        refined_kernel = self.compensate_kernel(kernel_para, factor, xrange)
        xmesh, ymesh = self.make_meshgrid(grid, size)
        xmesh = np.array(xmesh)
        ymesh = np.array(ymesh)
        
        kernel_sample_images = self.compute_sample_kernels(refined_kernel,xmesh, ymesh)
        
        print(kernel_sample_images.shape)
    
    def xy_main(self, kernel_para_dict, grid:list, size:list, refine_factor):
        kernel_para = self.preprocess_kernel_para(kernel_para_dict,grid, size)
        xrange = [0, size[0]*kernel_para.shape[0]]
        xmesh, ymesh, kernel_samples = self.compute_kernels(kernel_para, grid, size, xrange, refine_factor)
        np.save('siddon_kernel_xy_samples.npy',kernel_samples)
    

    def preprocess_z_para(original_para, grid, size):
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
        voxsize = size/grid
        kernel_Z = original_para
        a = np.array(kernel_Z[:, 3])
        uz = np.array((kernel_Z[:, 2]-math.ceil(grid[2]/2))*voxsize[2])
        sigmaz = np.array(kernel_Z[:, 1]*voxsize[2])
        
        
        a_neg = np.flipud(a[1:])
        uz_neg = -np.flipud(uz[1:])
        sigmaz_neg = np.flipud(sigmaz[1:])

        compen_len = int((grid[2]-len(kernel_Z)*2+1)/2)
        print(compen_len)
        a = np.hstack(([0.0]* compen_len, a_neg, a, [0.0]* compen_len))
        uz = np.hstack(([0.0]* compen_len,uz_neg, uz,[0.0]* compen_len))
        sigmaz = np.hstack(([1.0]* compen_len, sigmaz_neg, sigmaz,[1.0]* compen_len))
        
        return np.vstack((a, uz, sigmaz)).T
    
    def compute_z_sample_kernels(kernel_z_para, zmesh):
        nb_samples = len(kernel_z_para) 
        kernel_z = np.zeros((grid[2], grid[2]),dtype=np.float32)
        for k in range(nb_samples):
            a = kernel_z_para[k,0]
            uz = kernel_z_para[k,1]
            sigz = kernel_z_para[k,2]
            for i in range(grid[2]):
                kernel_z[i, k] = a*np.exp(-(zmesh[i]-uz)**2/(2*sigz**2))
        return kernel_z

    def compute_z_kernels(kernel_para, grid, size):
        sz = size[2]
        nz = grid[2]
        iz = sz/nz
        zmesh = np.linspace((-sz+iz)/2, (sz-iz)/2, nz)
        kernel_z = compute_z_sample_kernels(kernel_z_para, zmesh)
        
        return zmesh, kernel_z
    
    def z_main(kernel_para_dict, )
        