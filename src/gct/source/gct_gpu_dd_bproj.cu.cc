#define GOOGLE_CUDA 1
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "../include/gct_gpu_dd_bproj.cu.h"

void BackprojectionKernelLauncher(const float *proj,
							  const ImgInfo imgInfo,
							  const ProjInfo projInfo,
							  float *result){
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    struct cudaExtent extent = make_cudaExtent(projInfo.grid[0], projInfo.grid[1], 1);
	cudaTextureObject_t tex_proj = 0;
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;

	cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.addressMode[2] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    cudaArray *array_proj;
    cudaMalloc3DArray(&array_proj, &channelDesc, extent);
    cudaMemcpy3DParms copyParams = {0};
    copyParams.dstArray = array_proj;
	copyParams.extent = extent;
    copyParams.kind = cudaMemcpyDeviceToDevice;
    cudaPitchedPtr dp_proj = make_cudaPitchedPtr((void*) proj, projInfo.grid[0] * sizeof(float), projInfo.grid[0],  projInfo.grid[1]);
    copyParams.srcPtr = dp_proj;
    
    cudaMemcpy3D(&copyParams);
    resDesc.res.array.array = array_proj;
    // cudaTextureObject_t tex_proj = host_create_texture_object(d_proj, pp.nb, pp.na, 1);
    cudaCreateTextureObject(&tex_proj, &resDesc, &texDesc, NULL);

    const dim3 gridSize((imgInfo.grid[0] + BLOCKSIZE_X - 1) / BLOCKSIZE_X, (imgInfo.grid[1] + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y, (imgInfo.grid[2] + BLOCKSIZE_Z - 1) / BLOCKSIZE_Z);
    const dim3 blockSize(BLOCKSIZE_X,BLOCKSIZE_Y, BLOCKSIZE_Z);

	BackprojectionKernel<<<gridSize, blockSize>>>(tex_proj, projInfo.angle, projInfo.SAD,
    projInfo.SID, projInfo.size[0], projInfo.grid[0], projInfo.center[0], projInfo.size[1],
    projInfo.grid[1], projInfo.center[1], imgInfo.grid[0], imgInfo.grid[1], imgInfo.grid[2],
    result);
    cudaDeviceSynchronize();
    cudaFreeArray(array_proj);
    cudaDestroyTextureObject(tex_proj);                                  
	
}

__global__ void BackprojectionKernel(cudaTextureObject_t tex_proj, float angle, float SO, float SD, int na, int nb, float da, float db, float ai, float bi, int nx, int ny, int nz, float *img){
    int ix = BLOCKSIZE_X * blockIdx.x + threadIdx.x;
    int iy = BLOCKSIZE_Y * blockIdx.y + threadIdx.y;
    int iz = BLOCKSIZE_Z * blockIdx.z + threadIdx.z;
    if (ix >= nx || iy >= ny || iz >= nz)
        return;

    int id = ix + iy * nx + iz * nx * ny;
    img[id] = 0.0f;
	float sphi = __sinf(angle);
	float cphi = __cosf(angle);
	// float dd_voxel[3];
	float xc, yc, zc;
	xc = (float)ix - nx / 2 + 0.5f;
	yc = (float)iy - ny / 2 + 0.5f;
	zc = (float)iz - nz / 2 + 0.5f;

	// voxel boundary coordinates
	float xll, yll, zll, xlr, ylr, zlr, xrl, yrl, zrl, xrr, yrr, zrr, xt, yt, zt, xb, yb, zb;

	xll = +xc * cphi + yc * sphi - 0.5f;
    yll = -xc * sphi + yc * cphi - 0.5f;
    xrr = +xc * cphi + yc * sphi + 0.5f;
    yrr = -xc * sphi + yc * cphi + 0.5f;
    zll = zc; zrr = zc;
	xrl = +xc * cphi + yc * sphi + 0.5f;
    yrl = -xc * sphi + yc * cphi - 0.5f;
    xlr = +xc * cphi + yc * sphi - 0.5f;
    ylr = -xc * sphi + yc * cphi + 0.5f;
    zrl = zc; zlr = zc;
    xt = xc * cphi + yc * sphi;
    yt = -xc * sphi + yc * cphi;
    zt = zc + 0.5f;
    xb = xc * cphi + yc * sphi;
    yb = -xc * sphi + yc * cphi;
    zb = zc - 0.5f;

	// the coordinates of source and detector plane here are after rotation
	float ratio, all, bll, alr, blr, arl, brl, arr, brr, at, bt, ab, bb, a_max, a_min, b_max, b_min;
	// calculate a value for each boundary coordinates
	

	// the a and b here are all absolute positions from isocenter, which are on detector planes
	ratio = SD / (xll + SO);
	all = ratio * yll;
	bll = ratio * zll;
	ratio = SD / (xrr + SO);
	arr = ratio * yrr;
	brr = ratio * zrr;
	ratio = SD / (xlr + SO);
	alr = ratio * ylr;
	blr = ratio * zlr;
	ratio = SD / (xrl + SO);
	arl = ratio * yrl;
	brl = ratio * zrl;
	ratio = SD / (xt + SO);
	at = ratio * yt;
	bt = ratio * zt;
	ratio = SD / (xb + SO);
	ab = ratio * yb;
	bb = ratio * zb;

	// get the max and min values of all boundary projectors of voxel boundaries on detector plane
	a_max = MAX6(all ,arr, alr, arl, at, ab);
	a_min = MIN6(all ,arr, alr, arl, at, ab);
	b_max = MAX6(bll ,brr, blr, brl, bt, bb);
	b_min = MIN6(bll ,brr, blr, brl, bt, bb);

	// the related positions on detector plane from start points
	a_max = a_max / da - ai + 0.5f; //  now they are the detector coordinates
	a_min = a_min / da - ai + 0.5f;
	b_max = b_max / db - bi + 0.5f;
	b_min = b_min / db - bi + 0.5f;
	int a_ind_max = (int)floorf(a_max); 	
	int a_ind_min = (int)floorf(a_min); 
	int b_ind_max = (int)floorf(b_max); 
	int b_ind_min = (int)floorf(b_min); 

	float bin_bound_1, bin_bound_2, wa, wb;
	for (int ia = MAX(0, a_ind_min); ia < MIN(na, a_max); ia ++){

		bin_bound_1 = ia + 0.0f;
		bin_bound_2 = ia + 1.0f;
		
		wa = MIN(bin_bound_2, a_max) - MAX(bin_bound_1, a_min);// wa /= a_max - a_min;

		for (int ib = MAX(0, b_ind_min); ib < MIN(nb, b_max); ib ++){
			bin_bound_1 = ib + 0.0f;
			bin_bound_2 = ib + 1.0f;
			wb = MIN(bin_bound_2, b_max) - MAX(bin_bound_1, b_min);// wb /= b_max - b_min;


			img[id] += wa * wb * tex3D<float>(tex_proj, (ia + 0.5f), (ib + 0.5f), 0.5f);
		}		
	}
}

#endif
