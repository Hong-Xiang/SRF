#define GOOGLE_CUDA 1
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "../include/gct_gpu_dd_proj.cu.h"

void ProjectionKernelLauncher(const float *img,
							  const ImgInfo imgInfo,
							  const ProjInfo projInfo,
							  float *result){
	const dim3 gridSize((projInfo.grid[0] + BLOCKSIZE_X - 1) / BLOCKSIZE_X, (projInfo.grid[1] + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y, 1);
    const dim3 blockSize(BLOCKSIZE_X,BLOCKSIZE_Y, BLOCKSIZE_Z);
	ProjectionKernel<<<gridSize, blockSize>>>(img, projInfo.angle, projInfo.SAD,
    projInfo.SID, projInfo.size[0], projInfo.grid[0], projInfo.center[0], projInfo.size[1],
    projInfo.grid[1], projInfo.center[1], imgInfo.grid[0], imgInfo.grid[1], imgInfo.grid[2],
    result);
}

__global__ void ProjectionKernel(const float *img, float angle, float SO, float SD, float da, int na,
	float ai, float db, int nb, float bi, int nx, int ny, int nz, float *proj){
    int ia = BLOCKSIZE_X * blockIdx.x + threadIdx.x;
    int ib = BLOCKSIZE_Y * blockIdx.y + threadIdx.y;
    if (ia >= na || ib >= nb)
        return;
    int id = ia + ib * na;
    proj[id] = 0.0f;
    float x1, y1, z1, x2, y2, z2, x20, y20, cphi, sphi;
    cphi = (float)cosf(angle);
    sphi = (float)sinf(angle);
    
    x1 = -SO * cphi;
    y1 = -SO * sphi;
    z1 = 0.0f;
    x20 = SD - SO;
    y20 = (ia + ai) * da; // locate the detector cell center before any rotation
    x2 = x20 * cphi - y20 * sphi;
    y2 = x20 * sphi + y20 * cphi;
    z2 = (ib + bi) * db;
    float x21, y21, z21; // offset between source and detector center
    x21 = x2 - x1;
    y21 = y2 - y1;
    z21 = z2 - z1;

    // y - z plane, where ABS(x21) > ABS(y21)
    if (ABS(x21) > ABS(y21)){
    // if (ABS(cphi) > ABS(sphi)){
        float yi1, yi2, zi1, zi2;
        int Yi1, Yi2, Zi1, Zi2;
        // for each y - z plane, we calculate and add the contribution of related pixels
        for (int ix = 0; ix < nx; ix++){
            // calculate y indices of intersecting voxel candidates
            float xl, xr, yl, yr, ratio;
            float cyll, cylr, cyrl, cyrr, xc;
            xl = x21 - da / 2 * sphi;
            xr = x21 + da / 2 * sphi;
            yl = y21 - da / 2 * cphi;
            yr = y21 + da / 2 * cphi;
            xc = (float)ix + 0.5f - (float)nx / 2 - x1;
            
            ratio = yl / xl;
            cyll = ratio * xc + y1 + ny / 2;
            ratio = yl / xr;
            cylr = ratio * xc + y1 + ny / 2;
            ratio = yr / xl;
            cyrl = ratio * xc + y1 + ny / 2;
            ratio = yr / xr;
            cyrr = ratio * xc + y1 + ny / 2;

            yi1 = MIN4(cyll, cylr, cyrl, cyrr); Yi1 = (int)floorf(yi1);
            yi2 = MAX4(cyll, cylr, cyrl, cyrr); Yi2 = (int)floorf(yi2);

            float zl, zr, czl, czr;
            zl = z21 - db / 2;
            zr = z21 + db / 2;
            xc = (float)ix + 0.5f - (float)nx / 2 - x1 ;

            ratio = zl / x21;
            czl = ratio * xc + z1 + nz / 2;
            ratio = zr / x21;
            czr = ratio * xc + z1 + nz / 2;

            zi1 = MIN(czl, czr); Zi1 = (int)floorf(zi1);
            zi2 = MAX(czl, czr); Zi2 = (int)floorf(zi2);

            float wy, wz;

            for (int iy = MAX(0, Yi1); iy <= MIN(ny - 1, Yi2); iy++)
            {
                wy = MIN(iy + 1.0f, yi2) - MAX(iy + 0.0f, yi1); wy /= (yi2 - yi1);
                for (int iz = MAX(0, Zi1); iz <= MIN(nz - 1, Zi2); iz++)
                {
                    wz = MIN(iz + 1.0f, zi2) - MAX(iz + 0.0f, zi1); wz /= (zi2 - zi1);
                    proj[id] += img[ix + iy * nx + iz * nx * ny] * wy * wz;
                }                
            }        
        }
    }
    // x - z plane, where ABS(x21) <= ABS(y21)    
    else{
        float xi1, xi2, zi1, zi2;
        int Xi1, Xi2, Zi1, Zi2;
        // for each y - z plane, we calculate and add the contribution of related pixels
        for (int iy = 0; iy < ny; iy++){
            // calculate y indices of intersecting voxel candidates
            float yl, yr, xl, xr, ratio;
            float cxll, cxlr, cxrl, cxrr, yc;
            yl = y21 - da / 2 * cphi;
            yr = y21 + da / 2 * cphi;
            xl = x21 - da / 2 * sphi;
            xr = x21 + da / 2 * sphi;
            yc = (float)iy + 0.5f - (float)ny / 2 - y1;
            
            ratio = xl / yl;
            cxll = ratio * yc + x1 + nx / 2;
            ratio = xl / yr;
            cxlr = ratio * yc + x1 + nx / 2;
            ratio = xr / yl;
            cxrl = ratio * yc + x1 + nx / 2;
            ratio = xr / yr;
            cxrr = ratio * yc + x1 + nx / 2;

            xi1 = MIN4(cxll, cxlr, cxrl, cxrr); Xi1 = (int)floorf(xi1);
            xi2 = MAX4(cxll, cxlr, cxrl, cxrr); Xi2 = (int)floorf(xi2);

            float zl, zr, czl, czr;
            zl = z21 - db / 2;
            zr = z21 + db / 2;
            yc = (float)iy + 0.5f - (float)ny / 2 - y1;

            ratio = zl / y21;
            czl = ratio * yc + z1 + nz / 2;
            ratio = zr / y21;
            czr = ratio * yc + z1 + nz / 2;

            zi1 = MIN(czl, czr); Zi1 = (int)floorf(zi1);
            zi2 = MAX(czl, czr); Zi2 = (int)floorf(zi2);

            float wx, wz;

            for (int ix = MAX(0, Xi1); ix <= MIN(nx - 1, Xi2); ix++)
            {
                wx = MIN(ix + 1.0f, xi2) - MAX(ix + 0.0f, xi1); wx /= (xi2 - xi1);
                for (int iz = MAX(0, Zi1); iz <= MIN(nz - 1, Zi2); iz++)
                {
                    wz = MIN(iz + 1.0f, zi2) - MAX(iz + 0.0f, zi1); wz /= (zi2 - zi1);
                    proj[id] += img[ix + iy * nx + iz * nx * ny] * wx * wz;
                }                
            }        
        }            
    }
}
#endif
