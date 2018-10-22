#define GOOGLE_CUDA 1
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "cuda.h"
#include "cuda_runtime.h"
#include <ctime>
#include <cmath>
//
#define MAX(x, y) ((x) > (y)) ? (x) : (y)
#define MIN(x, y) ((x) < (y)) ? (x) : (y)
#define MIN3(x, y, z) MIN(MIN((x), (y)), (z))
#define ABS(x) ((x > 0) ? x : -(x))
#define EPS 0.000001f
#define EPS1 0
#define DELTA 0
#define PI 3.14159265359

const int BLOCKSIZE_X = 16;
const int BLOCKSIZE_Y = 16;
const int BLOCKSIZE_Z = 1;
struct RayCast{
    int iImg[3]; // integer indices in image of current ray casting
    float fImg[3]; // true indices in image of current ray casting
    float T;
    float deltaT[3];
    int subNext[3]; // 1 or -1 or 0, depends on direction sign
    bool inBuf[3];
    bool inBufAll;
    int boffs;
};

struct Block{
    int grid[3];
    float size[3];
    float center[3];
};

struct Ray{
    float start_point[3];
    float direction[3];
    float length;
    float min_t;
    float max_t;
    int block_diff;
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

    ray.direction[0] = dx / length;
    ray.direction[1] = dy / length;
    ray.direction[2] = dz / length;

    ray.min_t = 0;
    ray.max_t = length;
    float theta1 = atan2(y1, x1); 
    int iblock1 = (int)floor(theta1 / PI * 8) % 16;
    float theta2 = atan2(y2, x2);
    int iblock2 = (int)floor(theta2 / PI * 8) % 16;
    ray.block_diff = MIN(ABS(iblock1 - iblock2), 16 - ABS(iblock1 - iblock2));

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

/*
Function description:
This function compute the intersected interval between a 1D limit with a ray.
   Args:
    toplim: the top bound limit of the image in some axis (x, y, z).
    botlim: the bottom bound limit of the image in some axis. 
    start_point: the start point of ray.
    direction: the direction unit vector of the ray.
    max_t: max t value of ray smaller than the length of ray
    min_t: min t value of ray (larger than 0)
   Returns:
    flag: min_t < max_t
    
*/

__device__ bool
ClipRay2Img(const int comp, Ray &ray, const Block &imgbox)
{
    float start_point = ray.start_point[comp];
    float direction = ray.direction[comp];
    float t;
    float botlim = imgbox.center[comp] - 0.5f * imgbox.size[comp];
    float toplim = imgbox.center[comp] + 0.5f * imgbox.size[comp];
    float max_t = ray.max_t;
    float min_t = ray.min_t;
    if (direction > EPS)
    {
        t = (toplim - start_point) / direction;
        if (t < max_t)
            max_t = t;
        t = (botlim - start_point) / direction;
        if (t > min_t)
            min_t = t;
    }
    else if (direction < -EPS)
    {
        t = (botlim - start_point) / direction;
        if (t < max_t)
            max_t = t;
        t = (toplim - start_point) / direction;
        if (t > min_t)
            min_t = t;
    }
    else
    {
        if ((start_point < botlim) || (start_point > toplim))
            return false;
    }
    ray.min_t = min_t;
    ray.max_t = max_t;
    /* Check that all this clipping has not removed the interval. */
    return min_t < max_t;
}

/*
Function description:
   Args:
    box: the image block.
    ray: an lor ray.
   Returns:
    flag: if the ray passes through the image box.
*/
__device__ bool
IsThroughImage(const Block &box, Ray &ray)
{
    ray.min_t = 0;
    ray.max_t = ray.length;
    //get the bound od the box.
    // printf("IsThroughImage called!\n");
    /* Keep updating the new min and max t values for the line
     * clipping against each component one at a time */
    for (int i = 0; i < 3; i++)
    {
        if (!ClipRay2Img(i, ray, box))
        {
            return false;
        }
    }
    // printf("IsThroughImage called!\n");
    /* Check that all this clipping has not removed the interval,
        i.e. that the line intersects the bounding box. */
    return ray.min_t < ray.max_t;
}


__device__ void
SetupRayCastComponent(const Block &imgbox, const Ray &ray, int comp, RayCast &raycast)
{
    int v_count = imgbox.grid[comp]; // image component dimension
    float v_res = imgbox.size[comp] / imgbox.grid[comp]; // image component resolution
    
    /* compute component of point at intersection of volume bounding box */
    float direction = ray.direction[comp];
    float min_t = ray.min_t;
    raycast.T = min_t;
    float pt = min_t * direction + ray.start_point[comp]; // starting component point in image
    pt += imgbox.size[comp] / 2 - imgbox.center[comp];
    raycast.fImg[comp] = pt / v_res;
    raycast.iImg[comp] = int(raycast.fImg[comp]);
    float interval;
    if (direction > EPS){
        raycast.subNext[comp] = 1;
        interval = (1 + raycast.iImg[comp] - raycast.fImg[comp]) * v_res;
        raycast.deltaT[comp] = interval / direction;
    }
    else if (direction < -EPS){
        raycast.subNext[comp] = -1;
        interval = (raycast.fImg[comp] - raycast.iImg[comp]) * v_res;
        if (interval < EPS){
            interval += v_res;
            raycast.iImg[comp] -= 1;
        }
        raycast.deltaT[comp] = -interval / direction;
    }
    else{
        raycast.subNext[comp] = 0;
        raycast.deltaT[comp] = HUGE;
    }
    raycast.inBuf[comp] = raycast.iImg[comp] >= 0 && raycast.iImg[comp] < v_count;
    // if (comp == 0 and raycast.iImg[comp] > 100)
    //     {
    //         raycast.inBuf[comp] == false;
    //     }
}

/*
Function description:
This function compute the raycast of a ray with the image block.
   Args:
    imgbox: image block
    ray: the lor ray
    raycast: the raycast to be updated
   Returns:
*/
__device__ void
SetupRayCast(const Block &imgbox, Ray &ray, RayCast &raycast)
{
    SetupRayCastComponent(imgbox, ray, 0, raycast);
    SetupRayCastComponent(imgbox, ray, 1, raycast);
    SetupRayCastComponent(imgbox, ray, 2, raycast);

    int num_x = imgbox.grid[0];
    int num_y = imgbox.grid[1];
    raycast.inBufAll = raycast.inBuf[0] && raycast.inBuf[1] && raycast.inBuf[2];
    raycast.boffs = raycast.iImg[0] + num_x * (raycast.iImg[1] + num_y * raycast.iImg[2]);
}

__device__ void
UpdateRayCast(const Block & imgbox, const Ray &ray, RayCast &raycast)
{
    int i = -1;
    if (raycast.deltaT[0] <= raycast.deltaT[1] && raycast.deltaT[0] <= raycast.deltaT[2])
        i = 0;
    else if (raycast.deltaT[1] <= raycast.deltaT[2])
        i = 1;
    else
        i = 2;
    float delta;
    if (!DELTA)
        delta = raycast.deltaT[i];
    else
        delta = DELTA;
    raycast.T += delta;
    // float direction0 = ray.direction[i];
    float interval;
    for (int comp = 0; comp < 3; comp ++){
        float v_res = imgbox.size[comp] / imgbox.grid[comp]; // image component resolution
        float direction = ray.direction[comp];
        raycast.fImg[comp] += delta * direction / v_res;
        raycast.iImg[comp] = int(raycast.fImg[comp]);
        float pos = raycast.fImg[comp];
        if (direction > 0.0f){
            interval = (1 + raycast.iImg[comp] - raycast.fImg[comp]) * v_res;
            raycast.deltaT[comp] = interval / direction;
        }
        else if (direction < 0.0f){
            interval = (raycast.fImg[comp] - raycast.iImg[comp]) * v_res;
            if (interval < EPS){
                raycast.iImg[comp] -= 1;
                interval += v_res;
            }
            raycast.deltaT[comp] = -interval / direction;
        }
        else{
            raycast.deltaT[comp] = HUGE;
        }
        raycast.inBuf[comp] = raycast.iImg[comp] >= 0 && raycast.iImg[comp] < imgbox.grid[comp];
        // && raycast.T < ray.max_t - EPS1 && raycast.T > ray.min_t + EPS1;
        // if (comp == 0 and raycast.iImg[comp] > 100)
        // {
        //     raycast.inBuf[comp] == false;
        // }
    }
    raycast.inBufAll = raycast.inBuf[0] && raycast.inBuf[1] && raycast.inBuf[2];
    int num_x = imgbox.grid[0];
    int num_y = imgbox.grid[1];
    raycast.boffs = raycast.iImg[0] + num_x * (raycast.iImg[1] + num_y * raycast.iImg[2]);

}

__device__ void
RayTracing(const float &weight, const Ray & ray, RayCast &raycast, 
        const float *image_data, const Block &imgbox, float &vproj)
{
    while (raycast.inBufAll)
    {
        float delta;
        if (!DELTA)
            delta = MIN3(raycast.deltaT[0], raycast.deltaT[1], raycast.deltaT[2]);
        else
            delta = DELTA;        
        vproj += image_data[raycast.boffs] * delta * weight;
        UpdateRayCast(imgbox, ray, raycast);
    }
}

/*
Function description:
   Args:
 
   Returns:
*/
__device__ void
BackRayTracing(const float &weight, const Ray & ray, RayCast &raycast, 
const float vproj, const Block &imgbox, float *image_data)
{
    while (raycast.inBufAll)
    {
        float delta;
        if (!DELTA)
            delta = MIN3(raycast.deltaT[0], raycast.deltaT[1], raycast.deltaT[2]);
        else
            delta = DELTA;
        float value = delta * weight;
        if (vproj > 0)
            atomicAdd(image_data + raycast.boffs, value * vproj);
        UpdateRayCast(imgbox, ray, raycast);
    }
}


__global__ void
project(float * result, const float *vproj, const float *image_data,
        const int gx, const int gy, const int gz,
        const float cx, const float cy, const float cz,
        const float dx, const float dy, const float dz,
        const int bgx, const int bgy, const int bgz,
        const float bcx, const float bcy, const float bcz,
        const float bdx, const float bdy, const float bdz,
        const float inner_radius, const float outer_radius, const int nb_rings,
        const int nb_blocks_per_ring)
{
    int tid1 = blockIdx.x * BLOCKSIZE_X + threadIdx.x;
    int tid2 = blockIdx.y * BLOCKSIZE_Y + threadIdx.y;
    int nb_sensors = bgy * bgz * nb_rings * nb_blocks_per_ring;
    if (tid1 >= nb_sensors || tid2 >= nb_sensors)
        return;
    if (tid1 > tid2)
        return;
    int grid[3] = {gx, gy, gz};
    float size[3] = {dx * gx, dy * gy, dz * gz};
    float center[3] = {cx, cy, cz};
    int iring1 = tid1 / (bgy * bgz * nb_blocks_per_ring);
    int iblock1 = (tid1 - iring1 * bgy * bgz * nb_blocks_per_ring) / (bgy * bgz);
    int igz1 = tid1 % (bgy * bgz) / bgy;
    int igy1 = tid1 %  bgy;

    int iring2 = tid2 / (bgy * bgz * nb_blocks_per_ring);
    int iblock2 = (tid2 - iring2 * bgy * bgz * nb_blocks_per_ring) / (bgy * bgz);
    int igz2 = tid2 % (bgy * bgz) / bgy;
    int igy2 = tid2 %  bgy;

    if (iblock1 == iblock2)
        {result[tid1 + tid2 * nb_sensors] = 0.0f; return;}
    float x1, y1, z1, x2, y2, z2, tmpx, tmpy, theta;

    z1 = (igz1 + 0.5f) * bdz - 0.5f * bdz * bgz + bcz;
    z2 = (igz2 + 0.5f) * bdz - 0.5f * bdz * bgz + bcz;

    tmpx = (inner_radius + outer_radius) / 2;
    tmpy = (igy1 + 0.5f) * bdy - 0.5f * bdy * bgy + bcy;
    theta = iblock1 * PI * 2 / nb_blocks_per_ring;
    x1 = cos(theta) * tmpx - sin(theta) * tmpy;
    y1 = sin(theta) * tmpx + cos(theta) * tmpy;

    tmpy = (igy2 + 0.5f) * bdy - 0.5f * bdy * bgy + bcy;
    theta = iblock2 * PI * 2 / nb_blocks_per_ring;
    x2 = cos(theta) * tmpx - sin(theta) * tmpy;
    y2 = sin(theta) * tmpx + cos(theta) * tmpy;


    Ray ray;
    Block imgbox;
    // step1: create the ray and image block.
    CreateRay(x1, y1, z1, x2, y2, z2, ray);
    CreateBlock(grid, size, center, imgbox);
    // step2: judge if the ray pass through the image region.
    if (IsThroughImage(imgbox, ray))
    {
        // step3: cast the ray.
        RayCast raycast;
        SetupRayCast(imgbox, ray, raycast);
        float weight = 1 / ray.length;// / ray.length;

        // step4: raytracing the raycast and integrate the ray.
        RayTracing(weight, ray, raycast, image_data, imgbox, result[tid1 + tid2 * nb_sensors]);
    }
    else
    {
        result[tid1 + tid2 * nb_sensors] = 0.0f;
    }
    if (result[tid1 + tid2 * nb_sensors] > 0.1)
        result[tid1 + tid2 * nb_sensors] = vproj[tid1 + tid2 * nb_sensors] / result[tid1 + tid2 * nb_sensors];
    if (result[tid1 + tid2 * nb_sensors] > 10000000)
        result[tid1 + tid2 * nb_sensors] = 0.0f;

}

__global__ void
backproject(float *image_data, const float *vproj, 
        const int gx, const int gy, const int gz,
        const float cx, const float cy, const float cz,
        const float dx, const float dy, const float dz,
        const int bgx, const int bgy, const int bgz,
        const float bcx, const float bcy, const float bcz,
        const float bdx, const float bdy, const float bdz,
        const float inner_radius, const float outer_radius, const int nb_rings,
        const int nb_blocks_per_ring)
{
    int tid1 = blockIdx.x * BLOCKSIZE_X + threadIdx.x;
    int tid2 = blockIdx.y * BLOCKSIZE_Y + threadIdx.y;
    int nb_sensors = bgy * bgz * nb_rings * nb_blocks_per_ring;
    if (tid1 >= nb_sensors || tid2 >= nb_sensors)
        return;
    if (tid1 >= tid2)
        return;
    int grid[3] = {gx, gy, gz};
    float size[3] = {dx * gx, dy * gy, dz * gz};
    float center[3] = {cx, cy, cz};
    int iring1 = tid1 / (bgy * bgz * nb_blocks_per_ring);
    int iblock1 = (tid1 - iring1 * bgy * bgz * nb_blocks_per_ring) / (bgy * bgz);
    int igz1 = tid1 % (bgy * bgz) / bgy;
    int igy1 = tid1 %  bgy;

    int iring2 = tid2 / (bgy * bgz * nb_blocks_per_ring);
    int iblock2 = (tid2 - iring2 * bgy * bgz * nb_blocks_per_ring) / (bgy * bgz);
    int igz2 = tid2 % (bgy * bgz) / bgy;
    int igy2 = tid2 %  bgy;

    float x1, y1, z1, x2, y2, z2, tmpx, tmpy, theta;

    z1 = (igz1 + 0.5f) * bdz - 0.5f * bdz * bgz + bcz;
    z2 = (igz2 + 0.5f) * bdz - 0.5f * bdz * bgz + bcz;

    tmpx = (inner_radius + outer_radius) / 2;
    tmpy = (igy1 + 0.5f) * bdy - 0.5f * bdy * bgy + bcy;
    theta = iblock1 * PI * 2 / nb_blocks_per_ring;
    x1 = cos(theta) * tmpx - sin(theta) * tmpy;
    y1 = sin(theta) * tmpx + cos(theta) * tmpy;

    tmpy = (igy2 + 0.5f) * bdy - 0.5f * bdy * bgy + bcy;
    theta = iblock2 * PI * 2 / nb_blocks_per_ring;
    x2 = cos(theta) * tmpx - sin(theta) * tmpy;
    y2 = sin(theta) * tmpx + cos(theta) * tmpy;


    Ray ray;
    Block imgbox;
    // step1: create the ray and image block.
    CreateRay(x1, y1, z1, x2, y2, z2, ray);
    CreateBlock(grid, size, center, imgbox);
    // step2: judge if the ray pass through the image region.
    if (IsThroughImage(imgbox, ray))
    {
        // step3: cast the ray.
        RayCast raycast;
        SetupRayCast(imgbox, ray, raycast);
        float weight = 1 / ray.length;// / ray.length;

        // step4: raytracing the raycast and integrate the ray.
        BackRayTracing(weight, ray, raycast, vproj[tid1 + nb_sensors * tid2], imgbox, image_data);
    }
    else
    {}
}



void projection(float * result, const float *projection_value, const float *image,
                const int *grid, const float *center, const float *size,
                const int *block_grid, const float *block_center, const float *block_size,
                const float inner_radius, const float outer_radius, const int nb_rings,
                const int nb_blocks_per_ring, const float gap)

{    
    int grid_cpu[3];
    float center_cpu[3];
    float size_cpu[3];
    int block_grid_cpu[3];
    float block_center_cpu[3];
    float block_size_cpu[3];
    cudaMemcpy(grid_cpu, grid, 3 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(center_cpu, center, 3 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(size_cpu, size, 3 * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaMemcpy(block_grid_cpu, block_grid, 3 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(block_center_cpu, block_center, 3 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(block_size_cpu, block_size, 3 * sizeof(float), cudaMemcpyDeviceToHost);
    int gx = grid_cpu[0], gy = grid_cpu[1], gz = grid_cpu[2]; //number of meshes
    float cx = center_cpu[0], cy = center_cpu[1], cz = center_cpu[2]; // position of center
    float sx = size_cpu[0], sy = size_cpu[1], sz = size_cpu[2];
    // int bgx = block_grid_cpu[0], bgy = block_grid_cpu[1], bgz = block_grid_cpu[2]; //number of meshes
    // float bcx = block_center_cpu[0], bcy = block_center_cpu[1], bcz = block_center_cpu[2]; // position of center
    // float bsx = block_size_cpu[0], bsy = block_size_cpu[1], bsz = block_size_cpu[2];
    int bgx = 1; int bgy = 10; int bgz = 10;
    float bcx = 0.0f; float bcy = 0.0f; float bcz = 0.0f;
    float bsx = 20.0f; float bsy = 33.4f; float bsz = 33.4f;
    int nb_sensors = 1600;//bgy * bgz * nb_rings * nb_blocks_per_ring;
	const dim3 gridSize((nb_sensors + BLOCKSIZE_X - 1) / BLOCKSIZE_X, (nb_sensors + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y, 1);
    const dim3 blockSize(BLOCKSIZE_X,BLOCKSIZE_Y, BLOCKSIZE_Z);
    project<<<gridSize, blockSize>>>(result, projection_value, image, 
                                   gx, gy, gz, cx, cy, cz, sx / gx, sy / gy, sz / gz,
                                   bgx, bgy, bgz, bcx, bcy, bcz, bsx / bgx, bsy / bgy, bsz / bgz,
                                   inner_radius, outer_radius, 1, 16);
}


void backprojection(float *image, const float *projection_value, 
                const int *grid, const float *center, const float *size,
                const int *block_grid, const float *block_center, const float *block_size,
                const float inner_radius, const float outer_radius, const int nb_rings,
                const int nb_blocks_per_ring, const float gap)
{
    int grid_cpu[3];
    float center_cpu[3];
    float size_cpu[3];
    int block_grid_cpu[3];
    float block_center_cpu[3];
    float block_size_cpu[3];
    cudaMemcpy(grid_cpu, grid, 3 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(center_cpu, center, 3 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(size_cpu, size, 3 * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaMemcpy(block_grid_cpu, block_grid, 3 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(block_center_cpu, block_center, 3 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(block_size_cpu, block_size, 3 * sizeof(float), cudaMemcpyDeviceToHost);
    int gx = grid_cpu[0], gy = grid_cpu[1], gz = grid_cpu[2]; //number of meshes
    float cx = center_cpu[0], cy = center_cpu[1], cz = center_cpu[2]; // position of center
    float sx = size_cpu[0], sy = size_cpu[1], sz = size_cpu[2];
    // int bgx = block_grid_cpu[0], bgy = block_grid_cpu[1], bgz = block_grid_cpu[2]; //number of meshes
    // float bcx = block_center_cpu[0], bcy = block_center_cpu[1], bcz = block_center_cpu[2]; // position of center
    // float bsx = block_size_cpu[0], bsy = block_size_cpu[1], bsz = block_size_cpu[2];
    int bgx = 1; int bgy = 10; int bgz = 10;
    float bcx = 0.0f; float bcy = 0.0f; float bcz = 0.0f;
    float bsx = 20.0f; float bsy = 33.4f; float bsz = 33.4f;
    int nb_sensors = 1600;//bgy * bgz * nb_rings * nb_blocks_per_ring;
	const dim3 gridSize((nb_sensors + BLOCKSIZE_X - 1) / BLOCKSIZE_X, (nb_sensors + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y, 1);
    const dim3 blockSize(BLOCKSIZE_X,BLOCKSIZE_Y, BLOCKSIZE_Z);
    backproject<<<gridSize, blockSize>>>(image, projection_value, 
                                   gx, gy, gz, cx, cy, cz, sx / gx, sy / gy, sz / gz,
                                   bgx, bgy, bgz, bcx, bcy, bcz, bsx / bgx, bsy / bgy, bsz / bgz,
                                   inner_radius, outer_radius, 1, 16);
}

void mapsino(float *image, const float *projection_value, 
            const int *grid, const float *center, const float *size,
            const int *block_grid, const float *block_center, const float *block_size,
            const float inner_radius, const float outer_radius, const int nb_rings,
            const int nb_blocks_per_ring, const float gap)
{
    int grid_cpu[3];
    float center_cpu[3];
    float size_cpu[3];
    int block_grid_cpu[3];
    float block_center_cpu[3];
    float block_size_cpu[3];
    cudaMemcpy(grid_cpu, grid, 3 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(center_cpu, center, 3 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(size_cpu, size, 3 * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaMemcpy(block_grid_cpu, block_grid, 3 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(block_center_cpu, block_center, 3 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(block_size_cpu, block_size, 3 * sizeof(float), cudaMemcpyDeviceToHost);
    int gx = grid_cpu[0], gy = grid_cpu[1], gz = grid_cpu[2]; //number of meshes
    float cx = center_cpu[0], cy = center_cpu[1], cz = center_cpu[2]; // position of center
    float sx = size_cpu[0], sy = size_cpu[1], sz = size_cpu[2];
    int bgx = 1; int bgy = 10; int bgz = 10;
    float bcx = 0.0f; float bcy = 0.0f; float bcz = 0.0f;
    float bsx = 20.0f; float bsy = 33.4f; float bsz = 33.4f;
    // int bgx = block_grid_cpu[0], bgy = block_grid_cpu[1], bgz = block_grid_cpu[2]; //number of meshes
    // float bcx = block_center_cpu[0], bcy = block_center_cpu[1], bcz = block_center_cpu[2]; // position of center
    // float bsx = block_size_cpu[0], bsy = block_size_cpu[1], bsz = block_size_cpu[2];
    int nb_sensors = 1600;//bgy * bgz * nb_rings * nb_blocks_per_ring;
	const dim3 gridSize((nb_sensors + BLOCKSIZE_X - 1) / BLOCKSIZE_X, (nb_sensors + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y, 1);
    const dim3 blockSize(BLOCKSIZE_X,BLOCKSIZE_Y, BLOCKSIZE_Z);
    backproject<<<gridSize, blockSize>>>(image, projection_value, 
                                   gx, gy, gz, cx, cy, cz, sx / gx, sy / gy, sz / gz,
                                   bgx, bgy, bgz, bcx, bcy, bcz, bsx / bgx, bsy / bgy, bsz / bgz,
                                   inner_radius, outer_radius, 1, 16);
}


#endif