% cd /home/chengaoyu/WorkSpace/SRF/debug/exp_short/
% ls
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import math

root = './'

import scipy.io as scio
sigmaXY_file = root+'sigmaXY.mat'
dataXY =  scio.loadmat(sigmaXY_file)
# print(dataXY)
sigmaXY = dataXY['sigma']
# print(sigmaXY)
sigmaXY = np.array(sigmaXY)
# print(sigmaXY)
np.save('sigmaXY.npy', sigmaXY)

sigmaZ_file = root + 'sigmaZ.mat'
dataZ = scio.loadmat(sigmaZ_file)
sigmaZ = dataZ['sigma']
sigmaZ = np.array(sigmaZ)
print(sigmaZ)
np.save('sigmaZ.npy', sigmaZ)

# print(sigmaXY.shape)
print(sigmaZ.shape)

root = './'
it = 2
ss = 0
img = np.load(root+'result/exp_short_psf_{}_{}.npy'.format(it, ss)).T
# img = np.load(root+'exp_short_map.npy'.format(it, ss)).T
mask = img == 1e8
img[mask] = 0
img1 = np.load(root+'result/exp_short_200ps_{}_{}.npy'.format(it, ss)).T
# img1 = np.load(root+'exp_short_map684.npy'.format(it, ss)).T
mask = img1 == 1e8
img1[mask] = 0

img2 = np.load(root+'exp_short_corase_map.npy'.format(it, ss)).T
mask = img2 == 1e8
img2[mask] = 0

print(img.shape)
a = img.shape
imgs = np.sum(np.sum(img, axis=1),axis=1)
islice = np.argmax(imgs)
print("islice:",islice)
print('max value:', np.max(img))
print('min value:', np.min(img))
vmax = None
vmin = None


plt.figure(figsize=(15, 20))
plt.subplot(3,3,1)
plt.imshow(img[round(a[0]/2),:,:], vmin = vmin, vmax = vmax)
plt.subplot(3,3,2)
plt.imshow(img[:,round(a[1]/2),:], vmin = vmin, vmax = vmax)
plt.subplot(3,3,3)
plt.imshow(img[:,:,round(a[2]/2)], vmin = vmin, vmax = vmax)

plt.subplot(3,3,4)
plt.imshow(img1[round(a[0]/2),:,:], vmin = vmin, vmax = vmax)
plt.subplot(3,3,5)
plt.imshow(img1[:,round(a[1]/2),:], vmin = vmin, vmax = vmax)
plt.subplot(3,3,6)
plt.imshow(img1[:,:,round(a[2]/2)], vmin = vmin, vmax = vmax)
      
plt.subplot(3,3,7)
plt.imshow(img2[round(a[0]/2),:,:], vmin = vmin, vmax = vmax)
plt.subplot(3,3,8)
plt.imshow(img2[:,round(a[1]/2),:], vmin = vmin, vmax = vmax)
plt.subplot(3,3,9)
plt.imshow(img2[:,:,round(a[2]/2)], vmin = vmin, vmax = vmax)

plt.figure(figsize=(20,5))
plt.subplot(121)
plt.plot(np.sum(np.sum(img1,axis = 1),axis = 1))
plt.subplot(122)
plt.plot(np.sum(np.sum(img,axis = 1),axis = 1))

plt.figure(figsize=(20,5))
plt.subplot(121)
plt.plot(img1[208,97,:])
plt.subplot(122)
plt.plot(img[208,97,:])
# 重构成一个函数

# psf computing
def make_meshgrid(grid, size):
    '''
    '''
    sx, sy = size[0], size[1]
    nx, ny = grid[0], grid[1]
    ix, iy = sx/nx, sy/ny
    x = np.linspace((-sx+ix)/2, (sx-ix)/2, nx)
    y = np.linspace((-sy+iy)/2, (sy-iy)/2, ny)
    xv, yv = np.meshgrid(x,y, indexing = 'ij')
    return xv, yv

def make_polargrid(xmesh, ymesh):
    '''
    '''
#     rmesh = np.zeros(xmesh.shape)
#     pmesh = np.zeros(xmesh.shape)
    
    rmesh = np.sqrt(xmesh**2+ymesh**2)
    pmesh = np.arctan2(ymesh, xmesh)*180/np.pi
    return rmesh, pmesh



def compensate_kernel(kernel_para, factor, xrange):
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

def locate_kernel(sampled_kernels, xrange, r):
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

def rotate_kernel(kernel_image, angle):
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
    

def preprocess_kernel_para(origin_kernel, grid, size):
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
    
def compute_sample_kernels(kernel_para, xmesh, ymesh):
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

def compute_kernels(kernel_para, grid, size, xrange, factor):  
#     print(kernel_para)    
    refined_kernel = compensate_kernel(kernel_para, factor, xrange)
#     print(refined_kernel.shape)

    xmesh, ymesh = make_meshgrid(grid, size)
    xmesh = np.array(xmesh)
    ymesh = np.array(ymesh)
    
    kernel_sample_images = compute_sample_kernels(refined_kernel,xmesh, ymesh)
    
    print(kernel_sample_images.shape)
    
#     plt.figure()
#     plt.imshow(kernel_sample_images[:,:, 90])
    return xmesh, ymesh, kernel_sample_images

def show_kernel(original_kernel, rotated_kernel,show_range):
    origin = None
    kernel_image, kernel = original_kernel, rotated_kernel
    
    plt.figure(figsize = (10,10))
    vmax = None
    vmin = None
    index0 = np.array(np.where((kernel_image == np.max(kernel_image))))
    index1 = np.array(np.where((kernel == np.max(kernel))))
    if np.max(kernel) == 0 and np.max(kernel_image) == 0:
        index0 = np.array([np.floor(kernel_image.shape[0]/2), np.floor(kernel_image.shape[1]/2)])
        index1 = np.array([np.floor(kernel.shape[0]/2), np.floor(kernel.shape[1]/2)])
#     print(index0)
#     print(index1)
    origin_xs = int(index0[0])-show_range
    origin_xe = origin_xs+2*show_range
    origin_ys = int(index0[1])-show_range
    origin_ye = origin_ys+2*show_range
    rotated_xs = int(index1[0])-show_range
    rotated_xe = rotated_xs+2*show_range
    rotated_ys = int(index1[1])-show_range
    rotated_ye = rotated_ys+2*show_range
#     print(index0)
    plt.subplot(221)
    plt.imshow(kernel_image[origin_xs:origin_xe, origin_ys:origin_ye ], vmax= vmax, vmin = vmin, origin = origin)
    plt.subplot(222)
    plt.imshow(kernel[rotated_xs:rotated_xe, rotated_ys:rotated_ye ], vmax= vmax, vmin = vmin, origin = origin)
    plt.subplot(223)
    plt.imshow(kernel_image, vmax= vmax, vmin = vmin, origin = origin)
    plt.xlabel('y')
    plt.ylabel('x')
    plt.subplot(224)
    plt.imshow(kernel, vmax= vmax, vmin = vmin, origin = origin)
    plt.xlabel('y')
    plt.ylabel('x')
    print("sum value of origin kerenl:",np.sum(kernel_image))
    print("sum value of rotated kernel:", np.sum(kernel))


#compute the sample kernel images
kernel_para = np.load('./sigmaXY.npy')
grid = np.array([195, 195, 575])
size = np.array([666.9, 666.9, 1966.5])
kernel_para = preprocess_kernel_para(kernel_para, grid, size)
grid = grid*1
xrange = [0, 92*3.42]
factor = 7
xmesh, ymesh, kernel_samples = compute_kernels(kernel_para, grid, size, xrange, factor)
np.save('kernel_xy_samples.npy',kernel_samples)

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
#     print(kernel_Z[:,3])
    a = np.array(kernel_Z[:, 1])
    uz = np.array((kernel_Z[:, 3]-math.ceil(grid[2]/2))*voxsize[2])
    sigmaz = np.array(kernel_Z[:, 2]*voxsize[2])
    a_neg = np.flipud(a[1:])
    uz_neg = -np.flipud(uz[1:])
    sigmaz_neg = np.flipud(sigmaz[1:])
    
#     print(uz_neg)
#     print(uz)
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

kernel_z_para = np.load('./sigmaZ.npy')
# print(kernel_z_para)
nb_points = len(kernel_z_para)
grid = np.array([195, 195, 415])
size = np.array([666.9, 666.9, 1419.3])
zrange = [-3.42*(nb_points - 1), 3.42*(nb_points - 1)]
kernel_z_para = preprocess_z_para(kernel_z_para, grid, size)
# print(kernel_z_para[100:200,:])

zmesh, kernel_z_samples = compute_z_kernels(kernel_z_para, grid, size)

np.save('kernel_z_dense.npy', kernel_z_samples)

plt.figure(figsize = [10,10])
plt.imshow(kernel_z_samples)
plt.colorbar()
plt.figure(figsize =[15, 5])
islice = 10
plt.subplot(1,3,1)
plt.plot(kernel_z_samples[islice-6:islice+6,islice])
islice1 = 207
plt.subplot(1,3,2)
plt.plot(kernel_z_samples[islice1-6:islice1+6,islice1])
islice2 = 405
plt.subplot(1,3,3)
plt.plot(kernel_z_samples[islice2-6:islice2+6,islice2])
# print(kernel_samples)

# print(xmesh[98,98])
# print(ymesh[98,98])
# test_r =  rmesh[163,163]
# test_p =  pmesh[163,163]
# print(test_p)
# original_kernel = locate_kernel(kernel_samples, xrange, test_r)
# print(original_kernel.shape)
# kernel = rotate_kernel(original_kernel, test_p)
# show_kernel(original_kernel, kernel, 4)
# from scipy import sparse

# kernel = kernel/np.sum(kernel)
# mask = kernel < 1e-4
# kernel[mask] = 0.0
# kernel_flat = kernel.reshape(-1)
# spa_kernel = sparse.coo_matrix(kernel_flat)
# print(spa_kernel)
# # print(spa_kernel.data)
# print(np.sum(spa_kernel))
# import sys
# print(sys.getsizeof(spa_kernel))
# print(sys.getsizeof(kernel))

# print(spa_kernel.col.shape)
# print(spa_kernel.row.shape)

def make_xy_kernel(rmesh, pmesh, grid, kernel_samples):
    import time
    nx, ny = grid[0], grid[1]
    row = []
    col = []
    data = []
    start = time.time()
#     nx = 1
#     ny = 1
    for i in range(nx):
#         i = i + 97
        for j in range(ny):
#             j = j + 97
            irow = j + i*ny
            ir = rmesh[i, j]
            ip = pmesh[i, j]
            if(i == 0 and j== 0):
                print(ir)
            original_kernel = locate_kernel(kernel_samples, xrange, ir)
            kernel = rotate_kernel(original_kernel, ip)
            kernel_sum = np.sum(kernel)
            if kernel_sum < 1e-8:
                kernel_sum = 1
            kernel = kernel/kernel_sum
            mask = kernel < 1e-4
            kernel[mask] = 0.0
            kernel_flat = kernel.reshape((-1))
#             print('kernel_flat:\n',kernel_flat)
            from scipy import sparse
            spa_kernel = sparse.coo_matrix(kernel_flat)
            nb_ele = spa_kernel.data.size
            if nb_ele > 0:
#                 print('nb_ele:', nb_ele)
#                 print('irow:', irow)
                row = np.append(row, [irow]*nb_ele)
                col = np.append(col, spa_kernel.col)
                data = np.append(data, spa_kernel.data)
        end = time.time()
        diff = end -start
        print('used time: {} seconds.'.format(diff))
        print('time remain: {} seconds.'.format(diff/(i+1)*(nx-i-1)))
    return np.array([row, col, data])

def make_z_kernel( kernel_z_samples):
    nb_samples = len(kernel_z_samples)
    for i in range(nb_samples):
        irow = i
        kernel = kernel_z_samples[:, i]
        kernel_sum = np.sum(kernel)
        if kernel_sum < 1e-8:
            kernel_sum = 1
        kernel = kernel/kernel_sum
        mask = kernel < 1e-4
        kernel[mask] = 0.0
        kernel_z_samples[:,i] = kernel

from scipy import sparse

kernel_samples = np.load('kernel_xy_samples.npy')
rmesh ,pmesh = make_polargrid(xmesh, ymesh)
a = make_xy_kernel(rmesh, pmesh, grid, kernel_samples)
row = a[0]
col = a[1]
data = a[2]
print(row)
print(col)
print(data)
row = row.astype(int)
col = col.astype(int)
kernel_xy = sparse.coo_matrix((data, (row, col)), shape = (grid[0]*grid[1], grid[0]*grid[1]) ,dtype = np.float32)
print(kernel_xy.size)


# kernel_z = make_z_kernel(kernel_z_samples)
# kernel_z = sparse.coo_matrix(kernel_z_samples,shape = (grid[2], grid[2]))
# print(kernel_z.size)
# kernel_z = kernel_z.tocsr().tocoo()
import scipy.io as sio 
sio.savemat('psf_xy.mat', {'matrix':kernel_xy})
# sio.savemat('psf_z.mat', {'matrix':kernel_z})

# kernel_xy = sio.loadmat('psf_xy.mat')['matrix']
# kernel_z = sio.loadmat('psf_z.mat')['matrix']
print(kernel_xy.size)
# print(kernel_z.size)
# kernel = sparse.kron(kernel_xy, kernel_z)

# sparse.save_npz('psf_kernel.npz',kernel)

kernel_test = np.zeros((195,195))
for i in range(195*195):
    rowa = kernel_xy.getrow(i)
    kernel_test += (rowa.toarray().reshape(195,195))
plt.imshow(kernel_test)


rmesh ,tmesh = make_polargrid(xmesh, ymesh)
# print(xmesh[98,98])
# print(ymesh[98,98])
test_r =  rmesh[35,35]
test_t =  tmesh[35,35]
print(test_r)
print(test_t)
original_kernel = locate_kernel(kernel_samples, xrange, test_r)
kernel = rotate_kernel(original_kernel, test_t)
show_kernel(original_kernel, kernel, 5)

# plt.figure()
# plt.imshow(kernel)

kernel[kernel<1e-4] = 0
kernel = kernel.reshape(-1)
kernel_coo = sparse.coo_matrix(kernel)
print(kernel_coo)
print(kernel_coo.size)

from scipy import sparse

# A = sparse.coo_matrix([[1,2,3], [4,5,6], [7,8,9]])
# B = sparse.coo_matrix([[1,0],])
# print(A.shape)
# print(B.shape)

# C = sparse.kron(B,A).toarray()
# print('C', C.shape)
# E = C
# print('transpose E')
# print(E)
k_xy = sparse.diags([1], [0],shape = (195*195, 195*195), dtype = np.float32)
k_z = sparse.diags([1], [-15, ] ,shape = (415, 415), dtype = np.float32)
# k_xy_dense = k_xy.toarray()
k_z_dense = k_z.toarray()
# print(k_xy_dense.size)
print(k_z_dense.size)
# k_xy = k_xy.tocoo()
# kz = k_z.tocoo()

# k_xyz = sparse.kron(k_xy, k_z)
# print(k_xy)
# print(k_z)
# print(k_xyz)
# k_xyz = k_xyz.tocoo()
# print(k_xyz.size)
import scipy.io as sio 
sio.savemat('psf_xy.mat', {'matrix':k_xy})
# sio.savemat('exp_short_psf_matrix_z.mat', {'matrix':k_z})
# sio.savemat('exp_short_psf_matrix.mat', {'matrix':k_xyz})
# np.save('psf_z.npy', k_z_dense)
# sparse.save_npz('exp_short_psf_matrix.npz', k_xyz)

# plt.figure(figsize = [30,30])
# print(k_z)
# plt.subplot(131)
# plt.imshow(k_z_dense)
# print(k_xy)
# plt.subplot(132)
# plt.imshow(k_xy_dense)
# print(np.max(k_xy_dense))

aimport numpy as np
import matplotlib.pyplot as plt
def test_ndarray_order():
    a = np.arange(27)+1
    a = a.reshape([3,3,3])
#     a = np.array([[1,2,3],[4,5,6],[7,8,9]])
    print(a)
    print(a[1,2,2])
    print(a[0,0,:])
    
    plt.figure()
    plt.imshow(a[:,:,1])
    plt.colorbar()
    return a
a = test_ndarray_order()


import tensorflow as tf
a = tf.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])
b = a.values
with tf.Session() as sess:
    c = sess.run(b)
    print(c)


a = np.array([[1,2,3,4],[3,4,5,6],[5,6,7,8],[7,8,9,10]])
b = np.array([[1,2,3],[0,0,0],[3,2,1]])
x = np.arange(12)

x1 = x.reshape([4,3])
print('x1\n',x1)
c = np.kron(a,b)

x1 = x1.reshape((-1,1))
print('c:\n',c)
print('x1:\n', x1)
print('result:\n', c@x1)

# 
aT = a.T
xT = x.reshape((4,3)).T
print('xT:\n', xT)
result1 = (b@xT@aT).T
print('result1:\n', result1.reshape(-1,1))


a = np.array([[1,2,3,4],[3,4,5,6],[5,6,7,8],[7,8,9,10]])
x = np.arange(12)
print(a)
print(x)
x = x.reshape([4,3])
c = a@x
print(x)
print(c)
x = x.reshape([2,2,3])
print('x 3D:\n',x)
print(x[:,:,0])
x = x.reshape([4,3])
print('x 2D:\n',x)
print(x[:,0])


import scipy.io as sio
from scipy import sparse
kernel_xy = sparse.coo_matrix(sio.loadmat('psf_xy.mat')['matrix'])
# print(kernel_xy)
kernel_z = np.load('kernel_z_dense.npy')

effmap = np.load('exp_short_corase_map.npy')
mask = effmap >1e5
effmap[mask] = 0
grid = effmap.shape
effmap_reshaped = effmap.reshape((-1, grid[2]))

z_effmap = np.matmul(effmap_reshaped, kernel_z)

kernel_xy = sparse.coo_matrix((kernel_xy.data,(kernel_xy.col, kernel_xy.row)), 
                              shape = (grid[0]*grid[1], grid[0]*grid[1]), dtype = np.float32)

# print(kernel_xy)
psf_effmap = kernel_xy.dot(z_effmap)
z_effmap = z_effmap.reshape(grid)

psf_effmap = psf_effmap.reshape(grid)

mask = psf_effmap <1e-7
psf_effmap[mask] = 1e8

np.save('exp1.4_map_psf.npy', psf_effmap)

z_slice_num = 288 
plt.figure(figsize=[20,20])
plt.subplot(141)
plt.imshow(effmap[:,:, z_slice_num])

# print(kernel_z)
# plt.subplot(142)
# plt.imshow(kernel_xy.toarray())
# print(kernel_xy)
plt.subplot(143)
plt.imshow(z_effmap[:,:,z_slice_num])

plt.subplot(144)
plt.imshow(psf_effmap[:,:, z_slice_num])


z_slice_num1 = 144 
plt.figure(figsize=[20,20])
plt.subplot(141)
plt.imshow(effmap[:,:, z_slice_num1])

# print(kernel_z)
# plt.subplot(142)
# plt.imshow(kernel_z)
# print(kernel_xy)
plt.subplot(143)
plt.imshow(z_effmap[:,:,z_slice_num1])

plt.subplot(144)
plt.imshow(psf_effmap[:,:, z_slice_num1])


x_slice_num = 97 
plt.figure(figsize=[20,20])
plt.subplot(141)
plt.imshow(effmap[x_slice_num,:,: ])

# print(kernel_z)
# plt.subplot(142)
# plt.imshow(kernel_z)
# print(kernel_xy)
plt.subplot(143)
plt.imshow(z_effmap[x_slice_num,:,:])

plt.subplot(144)
plt.imshow(psf_effmap[x_slice_num,:,:])
