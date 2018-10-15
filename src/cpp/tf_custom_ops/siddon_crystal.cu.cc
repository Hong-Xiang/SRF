#define GOOGLE_CUDA 1
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "cuda.h"
#include "cuda_runtime.h"
#include <ctime>
#include <cmath>
//
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#define MIN3(x, y, z) MIN(MIN((x), (y)), (z))
#define ABS(x) ((x > 0) ? x : -(x))

#define EPS 0.00000001f
#define DELTA 0
#define SQRTROOT2 1.41421356237
const int BLOCKDIM = 1024;
struct RayCast{
    int dim; // 0 for a y-z plane and 1 for a x-z plane
    int iSlice;
    float center[3]; // intersection of ray with this plane
    float r_x;
    float r_y;
    float r_z;
    float cos_t;
    float sin_t;
};

struct Block{
    int grid[3];
    float size[3];
    float center[3];
};

struct Ray{
    float start_point[3];
    float end_point[3];    
    float direction[3];
    float length;
    float min_t;
    float max_t;
};

struct CrystalProjection{
    float radius = 109.0f;
    float r_xy = 1.67f;
    float r_z = 1.67f;
    float theta0[2]; // 0 for cos and 1 for sin
    float theta1[2];
};

struct TOF
{
    bool flag;
    // float limit2;
    float sigma2;
    float binsize;
};

__device__ void
CreateRay(float x1, float y1, float z1,
          float x2, float y2, float z2,
          Ray &ray)
{
    float dx = x2 - x1;
    float dy = y2 - y1;
    float dz = z2 - z1;
    float length = std::sqrt(dx * dx + dy * dy + dz * dz);
    ray.length = length;
    ray.start_point[0] = x1;
    ray.start_point[1] = y1;
    ray.start_point[2] = z1;
    ray.end_point[0] = x2;
    ray.end_point[1] = y2;
    ray.end_point[2] = z2;

    ray.direction[0] = dx / length;
    ray.direction[1] = dy / length;
    ray.direction[2] = dz / length;

    ray.min_t = 0;
    ray.max_t = length;
}

__device__ void
CreateCrystalProjection(const Ray & ray, CrystalProjection & cp)
{
    float x0 = ray.start_point[0], x1 = ray.start_point[1], y0 = ray.end_point[0] , y1 = ray.end_point[1];
    float R0 = std::sqrt(x0 * x0 + y0 * y0), R1 = std::sqrt(x1 * x1 + y1 * y1);
    cp.theta0[0] = x0 / R0;
    cp.theta0[1] = y0 / R0;
    cp.theta1[0] = x1 / R1;
    cp.theta1[1] = y1 / R1;
}

__device__ void
projectCrystal2Ray(const Ray & ray, CrystalProjection & cp)
{
    // projection in x-y plane
    // float r_xy = 3.34f;
    cp.r_xy = 1.67f / SQRTROOT2;
    
    float cos_theta_diff = cp.theta0[0] * cp.theta1[0] + cp.theta0[1] * cp.theta1[1];
    cp.r_xy *= std::sqrt(1.0f - cos_theta_diff);

    // projection along z axis
    float sin_theta_diff_half = std::sqrt((1 + cos_theta_diff) / 2.0f);
    cp.r_z = 1.67f / ray.length;
    cp.r_z *= 2 * cp.radius * sin_theta_diff_half;
}


__device__ void
CreateBlock(const int *grid, const float *size, const float *center, Block &imgbox)
{
    for (int i = 0; i < 3; i++)
    {
        imgbox.grid[i] = grid[i];
        imgbox.size[i] = size[i];
        imgbox.center[i] = center[i];
    }
}

__device__ void
UpdateRayCast(const Block &imgbox, const Ray &ray, const CrystalProjection & cp, int iSlice, RayCast &raycast)
{
    if (iSlice == -1)
    {
        if (ABS(ray.start_point[0] - ray.end_point[0]) >= ABS(ray.start_point[1] - ray.end_point[1]))
            raycast.dim = 0;
        else
            raycast.dim = 1;
        float x = ABS(ray.start_point[0] - ray.end_point[0]);
        float y = ABS(ray.start_point[1] - ray.end_point[1]);
        float R = std::sqrt(x * x + y * y);        
        raycast.cos_t = x / R;
        raycast.sin_t = y / R;
        float cos_z = std::sqrt(1 - ray.direction[2] * ray.direction[2]);
        raycast.r_x = 1.67f; //cp.r_xy / raycast.sin_t;
        raycast.r_y = 1.67f; //cp.r_xy / raycast.cos_t;
        raycast.r_z = 2;
        return;
    }
    
    int dim = raycast.dim;
    if (dim == 0) // intersected with a y-z plane
    {
        raycast.iSlice = iSlice;
        float xc = (0.5f + iSlice) * imgbox.size[0] / imgbox.grid[0] + imgbox.center[0] - imgbox.size[0] / 2;
        raycast.center[0] = xc;
        float alpha = (xc - ray.start_point[0]) / ray.direction[0];
        raycast.center[1] = alpha * ray.direction[1] + ray.start_point[1];
        raycast.center[2] = alpha * ray.direction[2] + ray.start_point[2];
    }
    else
    {
        raycast.iSlice = iSlice;
        float yc = (0.5f + iSlice) * imgbox.size[1] / imgbox.grid[1] + imgbox.center[1] - imgbox.size[1] / 2;
        raycast.center[1] = yc;
        float alpha = (yc - ray.start_point[1]) / ray.direction[1];
        raycast.center[0] = alpha * ray.direction[0] + ray.start_point[0];
        raycast.center[2] = alpha * ray.direction[2] + ray.start_point[2];
    }
}

__device__ void
RayTracing(RayCast &raycast, const Block &imgbox, const Ray &ray, const CrystalProjection &cp, const float *image_data, float &vproj)
{
    int nx = imgbox.grid[0];
    int ny = imgbox.grid[1];
    int nz = imgbox.grid[2];
    // vproj = 0.0f;
    if (ABS(ray.direction[0]) >= ABS(ray.direction[1]))
    {
        for (int ix = 0; ix < nx; ix++)
        {
            UpdateRayCast(imgbox, ray, cp, ix, raycast);

            float zbot = raycast.center[2] - raycast.r_z; zbot += imgbox.size[2] / 2 - imgbox.center[2]; zbot /= imgbox.size[2] / imgbox.grid[2];
            float ztop = raycast.center[2] + raycast.r_z; ztop += imgbox.size[2] / 2 - imgbox.center[2]; ztop /= imgbox.size[2] / imgbox.grid[2];
            if (zbot > ztop) {float tmp; tmp = ztop; ztop = zbot; zbot = tmp;}

            float ybot = raycast.center[1] - raycast.r_y; ybot += imgbox.size[1] / 2 - imgbox.center[1]; ybot /= imgbox.size[1] / imgbox.grid[1];
            float ytop = raycast.center[1] + raycast.r_y; ytop += imgbox.size[1] / 2 - imgbox.center[1]; ytop /= imgbox.size[1] / imgbox.grid[1];
            if (ybot > ytop) {float tmp; tmp = ytop; ytop = ybot; ybot = tmp;}        

            for (int iy = MAX(0, int(ybot)); iy <= MIN(ny - 1, int(ytop)); iy++)
            {
                float wy = MIN(iy + 1.0f, ytop) - MAX(iy + 0.0f, ybot); wy /= (ytop - ybot);
                for (int iz = MAX(0, int(zbot)); iz <= MIN(nz - 1, int(ztop)); iz++)
                {
                    float wz = MIN(iz + 1.0f, ztop) - MAX(iz + 0.0f, zbot); wz /= (ztop - zbot);
                    vproj += image_data[ix + iy * nx + iz * nx * ny]; // * wy * wz;
                }                
            }        
        }
    }
    else
    {
        for (int iy = 0; iy < ny; iy++)
        {
            UpdateRayCast(imgbox, ray, cp, iy, raycast);

            float zbot = raycast.center[2] - raycast.r_z; zbot += imgbox.size[2] / 2 - imgbox.center[2]; zbot /= imgbox.size[2] / imgbox.grid[2];
            float ztop = raycast.center[2] + raycast.r_z; ztop += imgbox.size[2] / 2 - imgbox.center[2]; ztop /= imgbox.size[2] / imgbox.grid[2];
            if (zbot > ztop) {float tmp; tmp = ztop; ztop = zbot; zbot = tmp;}

            float xbot = raycast.center[0] - raycast.r_x; xbot += imgbox.size[0] / 2 - imgbox.center[0]; xbot /= imgbox.size[0] / imgbox.grid[0];
            float xtop = raycast.center[0] + raycast.r_x; xtop += imgbox.size[0] / 2 - imgbox.center[0]; xtop /= imgbox.size[0] / imgbox.grid[0];
            if (xbot > xtop) {float tmp; tmp = xtop; xtop = xbot; xbot = tmp;}

            for (int ix = MAX(0, int(xbot)); ix <= MIN(nx - 1, int(xtop)); ix++)
            {
                float wx = MIN(ix + 1.0f, xtop) - MAX(ix + 0.0f, xbot); wx /= (xtop - xbot);
                for (int iz = MAX(0, int(zbot)); iz <= MIN(nz - 1, int(ztop)); iz++)
                {
                    float wz = MIN(iz + 1.0f, ztop) - MAX(iz + 0.0f, zbot); wz /= (ztop - zbot);
                    vproj += image_data[ix + iy * nx + iz * nx * ny]; // * wx * wz;
                }                
            }        
        }
    }
}

/*
Function description:
   Args:
 
   Returns:
*/
__device__ void
BackRayTracing(RayCast &raycast, const Block &imgbox, const Ray &ray, const CrystalProjection &cp, const float vproj, float *image_data)
{
    int nx = imgbox.grid[0];
    int ny = imgbox.grid[1];
    int nz = imgbox.grid[2];
    

    if (ABS(ray.direction[0]) >= ABS(ray.direction[1]))
    {
        for (int ix = 0; ix < nx; ix ++)
        {
            UpdateRayCast(imgbox, ray, cp, ix, raycast);

            float zbot = raycast.center[2] - raycast.r_z; zbot += imgbox.size[2] / 2 - imgbox.center[2]; zbot /= imgbox.size[2] / imgbox.grid[2];
            float ztop = raycast.center[2] + raycast.r_z; ztop += imgbox.size[2] / 2 - imgbox.center[2]; ztop /= imgbox.size[2] / imgbox.grid[2];
            if (zbot > ztop) {float tmp; tmp = ztop; ztop = zbot; zbot = tmp;}
            
            float ybot = raycast.center[1] - raycast.r_y; ybot += imgbox.size[1] / 2 - imgbox.center[1]; ybot /= imgbox.size[1] / imgbox.grid[1];
            float ytop = raycast.center[1] + raycast.r_y; ytop += imgbox.size[1] / 2 - imgbox.center[1]; ytop /= imgbox.size[1] / imgbox.grid[1];
            if (ybot > ytop) {float tmp; tmp = ytop; ytop = ybot; ybot = tmp;}        
            
            for (int iy = MAX(0, int(ybot)); iy <= MIN(ny - 1, int(ytop)); iy++)
            {
                float wy = MIN(iy + 1.0f, ytop) - MAX(iy + 0.0f, ybot);
                for (int iz = MAX(0, int(zbot)); iz <= MIN(nz - 1, int(ztop)); iz++)
                {
                    float wz = MIN(iz + 1.0f, ztop) - MAX(iz + 0.0f, zbot);
                    if (vproj > 0)
                        atomicAdd(image_data + ix + iy * nx + iz * nx * ny, 1.0f / vproj); // wy * wz / vproj);
                        // atomicAdd(image_data, 1.0f);
                }                
            }        
        }
    }
    else
    {
        for (int iy = 0; iy < ny; iy ++)
        {
            UpdateRayCast(imgbox, ray, cp, iy, raycast);

            float zbot = raycast.center[2] - raycast.r_z; zbot += imgbox.size[2] / 2 - imgbox.center[2]; zbot /= imgbox.size[2] / imgbox.grid[2];
            float ztop = raycast.center[2] + raycast.r_z; ztop += imgbox.size[2] / 2 - imgbox.center[2]; ztop /= imgbox.size[2] / imgbox.grid[2];
            if (zbot > ztop) {float tmp; tmp = ztop; ztop = zbot; zbot = tmp;}
            
            float xbot = raycast.center[0] - raycast.r_x; xbot += imgbox.size[0] / 2 - imgbox.center[0]; xbot /= imgbox.size[0] / imgbox.grid[0];
            float xtop = raycast.center[0] + raycast.r_x; xtop += imgbox.size[0] / 2 - imgbox.center[0]; xtop /= imgbox.size[0] / imgbox.grid[0];
            if (xbot > xtop) {float tmp; tmp = xtop; xtop = xbot; xbot = tmp;}
            
            for (int ix = MAX(0, int(xbot)); ix <= MIN(nx - 1, int(xtop)); ix++)
            {
                float wx = MIN(ix + 1.0f, xtop) - MAX(ix + 0.0f, xbot);
                for (int iz = MAX(0, int(zbot)); iz <= MIN(nz - 1, int(ztop)); iz++)
                {
                    float wz = MIN(iz + 1.0f, ztop) - MAX(iz + 0.0f, zbot);
                    if (vproj > 0)
                        atomicAdd(image_data + ix + iy * nx + iz * nx * ny, 1.0f / vproj); // wx * wz / vproj);
                        // atomicAdd(image_data, 1.0f);
                }
            }        
        }
    }
}

/*
Function description:
   Args:
 
   Returns:
*/
__device__ void
Map(RayCast &raycast, const Block &imgbox, const Ray &ray, const CrystalProjection &cp, const float vproj, float *image_data)
{
    int nx = imgbox.grid[0];
    int ny = imgbox.grid[1];
    int nz = imgbox.grid[2];
    
    if (ABS(ray.direction[0]) >= ABS(ray.direction[1]))
    {
        for (int ix = 0; ix < nx; ix ++)
        {
            UpdateRayCast(imgbox, ray, cp, ix, raycast);

            float zbot = raycast.center[2] - raycast.r_z; zbot += imgbox.size[2] / 2 - imgbox.center[2]; zbot /= imgbox.size[2] / imgbox.grid[2];
            float ztop = raycast.center[2] + raycast.r_z; ztop += imgbox.size[2] / 2 - imgbox.center[2]; ztop /= imgbox.size[2] / imgbox.grid[2];
            if (zbot > ztop) {float tmp; tmp = ztop; ztop = zbot; zbot = tmp;}

            float ybot = raycast.center[1] - raycast.r_y; ybot += imgbox.size[1] / 2 - imgbox.center[1]; ybot /= imgbox.size[1] / imgbox.grid[1];
            float ytop = raycast.center[1] + raycast.r_y; ytop += imgbox.size[1] / 2 - imgbox.center[1]; ytop /= imgbox.size[1] / imgbox.grid[1];
            if (ybot > ytop) {float tmp; tmp = ytop; ytop = ybot; ybot = tmp;}
            // if (ybot >= 0 and ybot < ny)
            // {
            //     if (zbot >= 0 and zbot < nz)
            //         atomicAdd(image_data + ix + int(ybot) * nx + int(zbot) * nx * ny, 1); // wy * wz / vproj);
            // }
            // continue;
            for (int iy = MAX(0, int(ybot)); iy <= MIN(ny - 1, int(ytop)); iy++)
            {
                float wy = MIN(iy + 1.0f, ytop) - MAX(iy + 0.0f, ybot);
                for (int iz = MAX(0, int(zbot)); iz <= MIN(nz - 1, int(ztop)); iz++)
                {
                    float wz = MIN(iz + 1.0f, ztop) - MAX(iz + 0.0f, zbot);
                    if (vproj > 0)
                        atomicAdd(image_data + ix + iy * nx + iz * nx * ny, 1.0f); // / vproj); 
                        // atomicAdd(image_data, 1.0f);
                }                
            }        
        }

    }
    else
    {
        for (int iy = 0; iy < ny; iy ++)
        {
            UpdateRayCast(imgbox, ray, cp, iy, raycast);

            float zbot = raycast.center[2] - raycast.r_z; zbot += imgbox.size[2] / 2 - imgbox.center[2]; zbot /= imgbox.size[2] / imgbox.grid[2];
            float ztop = raycast.center[2] + raycast.r_z; ztop += imgbox.size[2] / 2 - imgbox.center[2]; ztop /= imgbox.size[2] / imgbox.grid[2];
            if (zbot > ztop) {float tmp; tmp = ztop; ztop = zbot; zbot = tmp;}

            float xbot = raycast.center[0] - raycast.r_x; xbot += imgbox.size[0] / 2 - imgbox.center[0]; xbot /= imgbox.size[0] / imgbox.grid[0];
            float xtop = raycast.center[0] + raycast.r_x; xtop += imgbox.size[0] / 2 - imgbox.center[0]; xtop /= imgbox.size[0] / imgbox.grid[0];
            if (xbot > xtop) {float tmp; tmp = xtop; xtop = xbot; xbot = tmp;}
            // if (xbot >= 0 and xbot < nx)
            // {
            //     if (zbot >= 0 and zbot < nz)
            //         atomicAdd(image_data + int(xbot) + iy * nx + int(zbot) * nx * ny, 1); // wy * wz / vproj);
            // }

            // continue;
            for (int ix = MAX(0, int(xbot)); ix <= MIN(nx - 1, int(xtop)); ix++)
            {
                float wx = MIN(ix + 1.0f, xtop) - MAX(ix + 0.0f, xbot);
                for (int iz = MAX(0, int(zbot)); iz <= MIN(nz - 1, int(ztop)); iz++)
                {
                    float wz = MIN(iz + 1.0f, ztop) - MAX(iz + 0.0f, zbot);
                    if (vproj > 0)
                        atomicAdd(image_data + ix + iy * nx + iz * nx * ny, 1.0f);
                        // atomicAdd(image_data, 1.0f);
                }                
            }        
        }
    }
}

/*
Function description:
This function do the paralell computing of lor projection.
   Args:
 
   Returns:
*/
__global__ void
project(const float *x1, const float *y1, const float *z1,
        const float *x2, const float *y2, const float *z2,
        const float *tof_t, const float tof_bin, const float tof_sigma2,
        const int gx, const int gy, const int gz,
        const float cx, const float cy, const float cz,
        const float sx, const float sy, const float sz,
        const int num_events, const float *image_data, float *vproj)
{
    int tid = blockIdx.x * BLOCKDIM + threadIdx.x;
    if (tid >= num_events)
        return;
    int grid[3] = {gx, gy, gz};
    float size[3] = {sx, sy, sz};
    float center[3] = {cx, cy, cz};
    Ray ray;
    Block imgbox;
    CrystalProjection cp;
    // step1: create the ray and image block.
    CreateRay(x1[tid], y1[tid], z1[tid], x2[tid], y2[tid], z2[tid], ray);
    CreateBlock(grid, size, center, imgbox);
    CreateCrystalProjection(ray, cp);
    projectCrystal2Ray(ray, cp);
    RayCast raycast;
    UpdateRayCast(imgbox, ray, cp, -1, raycast);
    RayTracing(raycast, imgbox, ray, cp, image_data, vproj[tid]);
}

/*
Function description:
This function do the paralell computing of lor backprojection.
   Args:
 
   Returns:
*/
__global__ void
backproject(const float *x1, const float *y1, const float *z1,
            const float *x2, const float *y2, const float *z2,
            const float *tof_t,
            const float tof_bin, const float tof_sigma2,
            const int gx, const int gy, const int gz,
            const float cx, const float cy, const float cz,
            const float sx, const float sy, const float sz,
            const int num_events,
            const float *vproj, float *image_data)
{
    int tid = blockIdx.x * BLOCKDIM + threadIdx.x;
    if (tid >= num_events)
        return;
    int grid[3] = {gx, gy, gz};
    float size[3] = {sx, sy, sz};
    float center[3] = {cx, cy, cz};

    Ray ray;
    Block imgbox;
    CrystalProjection cp;

    // step1: create the ray and image block.
    CreateRay(x1[tid], y1[tid], z1[tid], x2[tid], y2[tid], z2[tid], ray);
    CreateBlock(grid, size, center, imgbox);
    CreateCrystalProjection(ray, cp);
    projectCrystal2Ray(ray, cp);
    RayCast raycast;
    UpdateRayCast(imgbox, ray, cp, -1, raycast);
    BackRayTracing(raycast, imgbox, ray, cp, vproj[tid], image_data);
}

__global__ void
mapping(const float *x1, const float *y1, const float *z1,
        const float *x2, const float *y2, const float *z2,
        const int gx, const int gy, const int gz,
        const float cx, const float cy, const float cz,
        const float sx, const float sy, const float sz,
        const int num_events,
        const float *vproj, float *image_data)
{
    int tid = blockIdx.x * BLOCKDIM + threadIdx.x;
    if (tid >= num_events)
        return;
    int grid[3] = {gx, gy, gz};
    float size[3] = {sx, sy, sz};
    float center[3] = {cx, cy, cz};

    Ray ray;
    Block imgbox;
    CrystalProjection cp;

    // step1: create the ray and image block.
    CreateRay(x1[tid], y1[tid], z1[tid], x2[tid], y2[tid], z2[tid], ray);
    float weight = 1 / ray.length / ray.length;
    CreateBlock(grid, size, center, imgbox);
    CreateCrystalProjection(ray, cp);
    projectCrystal2Ray(ray, cp);
    RayCast raycast;
    UpdateRayCast(imgbox, ray, cp, -1, raycast);
    BackRayTracing(raycast, imgbox, ray, cp, vproj[tid], image_data);
}


void projection(const float *x1, const float *y1, const float *z1,
                const float *x2, const float *y2, const float *z2,
                const float *tof_t, float *vproj,
                const int *grid, const float *center, const float *size,
                const float tof_bin, const float tof_sigma2,
                const float *image, const int num_events)
{
    
    int grid_cpu[3];
    float center_cpu[3];
    float size_cpu[3];
    cudaMemcpy(grid_cpu, grid, 3 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(center_cpu, center, 3 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(size_cpu, size, 3 * sizeof(float), cudaMemcpyDeviceToHost);
    int gx = grid_cpu[0], gy = grid_cpu[1], gz = grid_cpu[2]; //number of meshes
    float cx = center_cpu[0], cy = center_cpu[1], cz = center_cpu[2]; // position of center
    float sx = size_cpu[0], sy = size_cpu[1], sz = size_cpu[2];
    int GRIDDIM = num_events / BLOCKDIM + 1;
    project<<<GRIDDIM, BLOCKDIM>>>(x1, y1, z1,
                                   x2, y2, z2, 
                                   tof_t, tof_bin, tof_sigma2,
                                   gx, gy, gz, cx, cy, cz, sx, sy, sz,
                                   num_events, image, vproj);
}


void backprojection(const float *x1, const float *y1, const float *z1,
                    const float *x2, const float *y2, const float *z2,
                    const float *tof_t, const float *vproj,
                    const int *grid, const float *center, const float *size,
                    const float tof_bin, const float tof_sigma2,
                    float *image, const int num_events)
{
    int grid_cpu[3];
    float center_cpu[3];
    float size_cpu[3];
    cudaMemcpy(grid_cpu, grid, 3 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(center_cpu, center, 3 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(size_cpu, size, 3 * sizeof(float), cudaMemcpyDeviceToHost);
    int gx = grid_cpu[0], gy = grid_cpu[1], gz = grid_cpu[2]; //number of meshes
    float cx = center_cpu[0], cy = center_cpu[1], cz = center_cpu[2]; // position of center
    float sx = size_cpu[0], sy = size_cpu[1], sz = size_cpu[2];
    int GRIDDIM = num_events / BLOCKDIM + 1;
    backproject<<<GRIDDIM, BLOCKDIM>>>(x1, y1, z1,
                                       x2, y2, z2,
                                       tof_t, tof_bin, tof_sigma2,
                                       gx, gy, gz, cx, cy, cz, sx, sy, sz,
                                       num_events, vproj, image);
}


void maplors(const float *x1, const float *y1, const float *z1,
             const float *x2, const float *y2, const float *z2,
             const float *vproj,
             const int *grid, const float *center, const float *size,
             float *image, const int num_events)
{
    int grid_cpu[3];
    float center_cpu[3];
    float size_cpu[3];
    cudaMemcpy(grid_cpu, grid, 3 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(center_cpu, center, 3 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(size_cpu, size, 3 * sizeof(float), cudaMemcpyDeviceToHost);

    int gx = grid_cpu[0], gy = grid_cpu[1], gz = grid_cpu[2]; //number of meshes
    float cx = center_cpu[0], cy = center_cpu[1], cz = center_cpu[2]; // position of center
    float sx = size_cpu[0], sy = size_cpu[1], sz = size_cpu[2];

    int GRIDDIM = num_events / BLOCKDIM + 1;
    mapping<<<GRIDDIM, BLOCKDIM>>>(x1, y1, z1, 
                                   x2, y2, z2,
                                   gx, gy, gz, cx, cy, cz, sx, sy, sz,
                                   num_events, vproj, image);
}


#endif