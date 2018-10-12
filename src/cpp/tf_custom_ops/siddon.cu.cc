#define GOOGLE_CUDA 1
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "cuda.h"
#include "cuda_runtime.h"
#include <ctime>
#include <cmath>
//

const int GRIDDIM = 32;
const int BLOCKDIM = 1024;

struct RayCast
{
    int boffs;
    float nextT[3];
    float deltaT[3];
    int deltaBuf[3];
    int inBuf[3];
};
struct Block
{
    int grid[3];
    float size[3];
    float center[3];
};

struct Ray
{
    float start_point[3];
    float direction[3];
    float length;
    float min_t;
    float max_t;
};

struct TOF
{
    bool flag;
    // float limit2;
    float sigma2;
    float binsize;
};

/*
Function description:
This function creates a ray struct object with the input two points.
   Args:
    x1, y1, z1: the start point.
    x2, y2, z2: the end point.
 
   Returns:
    ray: the created ray.
   
   Note:
    the initialized min_t and max_t is 0 and ray length.
*/
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
}

/*
Function description:
This function creates a block struct object for the image geometry.
   Args:
    grid: the mesh grid of block.
    size: block region size.
    center: center position of the block.
   Returns:
    imgbox: the created block object.
*/
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
    p0: the start point of ray.
    delta_p: the direction unit vector of the ray.
    tmax: max t value of ray ï¼ˆsmaller than the length of rayï¼?
    tmin: min t value of ray (larger than 0)
   Returns:
    flag: tmin < tmax
    
*/
__device__ bool
ClipRay(float toplim, float botlim, int comp, Ray &ray)
{
    float p0 = ray.start_point[comp];
    float delta_p = ray.direction[comp];
    float tmax = ray.max_t;
    float tmin = ray.min_t;
    float t;
    if (delta_p > 0.0)
    {
        t = (toplim - p0) / delta_p;
        if (t < tmax)
            tmax = t;
        t = (botlim - p0) / delta_p;
        if (t > tmin)
            tmin = t;
    }
    else if (delta_p < 0.0)
    {
        t = (botlim - p0) / delta_p;
        if (t < tmax)
            tmax = t;
        t = (toplim - p0) / delta_p;
        if (t > tmin)
            tmin = t;
    }
    else
    {
        if ((p0 < botlim) || (p0 > toplim))
            return false;
    }
    ray.min_t = tmin;
    ray.max_t = tmax;
    /* Check that all this clipping has not removed the interval. */
    return tmin < tmax;
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
        if (!ClipRay(box.center[i] + box.size[i] / 2, box.center[i] - box.size[i] / 2, i, ray))
        {
            return false;
        }
    }
    // printf("IsThroughImage called!\n");
    /* Check that all this clipping has not removed the interval,
        i.e. that the line intersects the bounding box. */
    return ray.min_t < ray.max_t;
}

/*
Function description:
This function comute the parameter of a raycast in some axis.
   Args:
    imgbox: image box
    ray: current lor ray
    comp: axis index
    raycast: the raycast to be updated.
   Returns:
*/

__device__ void
SetupRayCastComponent(const Block &imgbox, const Ray &ray, int comp, RayCast &raycast, int &pos)
{
    int v_count;
    float p0, r_pdir, pt;
    float v_res;
    float mint;
    /* compute component of point at intersection of volume bounding box */
    p0 = ray.start_point[comp];
    r_pdir = ray.direction[comp];
    mint = ray.min_t;

    pt = mint * r_pdir + p0;

    /* get local copies of resolution, and dimension */
    v_res = imgbox.size[comp] / imgbox.grid[comp];
    v_count = imgbox.grid[comp];

    if (r_pdir > 0.0)
    {

        /* going to the right, so round down */

        pos = pt / v_res;
        raycast.nextT[comp] = ((pos + 1) * v_res - pt) / r_pdir;
        raycast.deltaT[comp] = v_res / r_pdir;
        raycast.deltaBuf[comp] = 1;
        raycast.inBuf[comp] = v_count - pos;
    }
    else if (r_pdir < 0.0)
    {

        /* going to the left, so round up and subtract 1 */
        pos = v_count - 1 - (int)(v_count - pt / v_res);
        raycast.nextT[comp] = (pos * v_res - pt) / r_pdir;
        raycast.deltaT[comp] = -v_res / r_pdir;
        raycast.deltaBuf[comp] = -1;
        raycast.inBuf[comp] = pos + 1;
    }
    else
    {
        pos = pt / v_res;
        raycast.nextT[comp] = HUGE;
        raycast.deltaT[comp] = HUGE;
        raycast.deltaBuf[comp] = 0;
        raycast.inBuf[comp] = 1;
    }

    /* get the correct spacing for the buffer pointer changes */

    for (int i = comp - 1; i >= 0; i--)
        raycast.deltaBuf[comp] *= imgbox.grid[i];
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
    // translate the origin to the corner of the image volume
    for (int i = 0; i < 3; i++)
    {
        ray.start_point[i] = ray.start_point[i] - (imgbox.center[i] - imgbox.size[i] / 2);
    }

    int pos_x = 0;
    int pos_y = 0;
    int pos_z = 0;
    SetupRayCastComponent(imgbox, ray, 0, raycast, pos_x);
    SetupRayCastComponent(imgbox, ray, 1, raycast, pos_y);
    SetupRayCastComponent(imgbox, ray, 2, raycast, pos_z);

    int num_x = imgbox.grid[0];
    int num_y = imgbox.grid[1];

    // int num_z = imgbox.grid[2];
    /* return buffer offset from first voxel along ray */
    raycast.boffs = pos_x + num_x * (pos_y + num_y * pos_z);
    // printf("boffs: %d \n", raycast.boffs);
}

/*
Function description:
This function integrate the raycast 
   Args:
 
   Returns:
*/
__device__ void
RayTracing(const TOF &tof_info, const float t_TOF, const float &weight,
           RayCast &raycast, const float *image_data, float &projection_value)
{
    RayCast &rc = raycast; /* local copy of raycast */
    float last_t = 0.0;    /* initialize last distance */
    float l_len = 0.0;
    int stillin = rc.inBuf[0] * rc.inBuf[1] * rc.inBuf[2]; /* make a non-zero test value unless we are already out */
    int i = -1;
    while (stillin)
    {
        /* look for next intersection: smallest component of nextT */
        i = (rc.nextT[0] <= rc.nextT[1]) ? ((rc.nextT[0] <= rc.nextT[2]) ? 0 : 2) : ((rc.nextT[1] <= rc.nextT[2]) ? 1 : 2);
        l_len = rc.nextT[i] - last_t;
        int mi = rc.boffs;
        if (tof_info.flag)
        {                                // use TOF information.
            float bs = tof_info.binsize; // get the binsize of TOF.
            float t = t_TOF - 0.5 * (rc.nextT[i] + last_t);
            float sigma2 = tof_info.sigma2 + (l_len * l_len + bs * bs) / 12.0;
            float t2_by_sigma2 = t * t / sigma2;
            if (t2_by_sigma2 < 9)
            { // the voxel is valid in the TOF range.
                projection_value += image_data[mi] * weight * l_len * tof_info.binsize * exp(-0.5 * t2_by_sigma2) / sqrt(2.0 * M_PI * sigma2);
            }
        }
        else //no TOF information.
        {
            projection_value += image_data[mi] * weight * l_len;
        }

        /* update */
        last_t = rc.nextT[i];        // set last distance for next pass
        rc.nextT[i] += rc.deltaT[i]; // set next intersection distance
        rc.boffs += rc.deltaBuf[i];  // buffer offset for next voxel
        stillin = rc.inBuf[i] - 1;
        rc.inBuf[i] = stillin; // if we go out this goes to zero
    }
}

/*
Function description:
   Args:
 
   Returns:
*/
__device__ void
BackRayTracing(const TOF &tof_info, const float t_TOF, const float &weight,
               RayCast &raycast, const float projection_value, float *image_data)
{
    RayCast &rc = raycast; /* local copy of raycast */
    float last_t = 0.0;    /* initialize last distance */
    float l_len = 0.0;
    int stillin = rc.inBuf[0] * rc.inBuf[1] * rc.inBuf[2]; /* make a non-zero test value unless we are already out */
    int i = -1;
    while (stillin)
    {
        /* look for next intersection: smallest component of nextT */
        i = (rc.nextT[0] <= rc.nextT[1]) ? ((rc.nextT[0] <= rc.nextT[2]) ? 0 : 2) : ((rc.nextT[1] <= rc.nextT[2]) ? 1 : 2);
        l_len = rc.nextT[i] - last_t;
        int mi = rc.boffs;
        if (tof_info.flag)
        {                                // use TOF information.
            float bs = tof_info.binsize; // get the binsize of TOF.
            float t = t_TOF - 0.5 * (rc.nextT[i] + last_t);
            float sigma2 = tof_info.sigma2 + (l_len * l_len + bs * bs) / 12.0;
            float t2_by_sigma2 = t * t / sigma2;
            if (t2_by_sigma2 < 9.0)
            { // the voxel is valid in the TOF range.
                float value = weight * l_len * tof_info.binsize * exp(-0.5 * t2_by_sigma2) / sqrt(2.0 * M_PI * sigma2);
                if (projection_value > 0)
                    atomicAdd(image_data + mi, value / (projection_value));
            }
        }
        else //no TOF information.
        {
            float value = weight * l_len;
            if (projection_value > 0)
                atomicAdd(image_data + mi, value / (projection_value));
        }

        /* update */
        last_t = rc.nextT[i];        // set last distance for next pass
        rc.nextT[i] += rc.deltaT[i]; // set next intersection distance
        rc.boffs += rc.deltaBuf[i];  // buffer offset for next voxel
        stillin = rc.inBuf[i] - 1;
        rc.inBuf[i] = stillin; // if we go out this goes to zero
    }
}

/*
Function description:
   Args:
 
   Returns:
*/
__device__ void
Map(const float &weight, RayCast &raycast, const float projection_value, float *image_data)
{
    RayCast &rc = raycast; /* local copy of raycast */
    float last_t = 0.0;    /* initialize last distance */
    float l_len = 0.0;
    int stillin = rc.inBuf[0] * rc.inBuf[1] * rc.inBuf[2]; /* make a non-zero test value unless we are already out */
    int i = -1;
    while (stillin)
    {
        /* look for next intersection: smallest component of nextT */
        i = (rc.nextT[0] <= rc.nextT[1]) ? ((rc.nextT[0] <= rc.nextT[2]) ? 0 : 2) : ((rc.nextT[1] <= rc.nextT[2]) ? 1 : 2);
        l_len = rc.nextT[i] - last_t;
        int mi = rc.boffs;
        float value = weight * l_len;

        /* update */
        last_t = rc.nextT[i];        // set last distance for next pass
        rc.nextT[i] += rc.deltaT[i]; // set next intersection distance
        rc.boffs += rc.deltaBuf[i];  // buffer offset for next voxel
        stillin = rc.inBuf[i] - 1;
        rc.inBuf[i] = stillin; // if we go out this goes to zero
        // printf("projection_value: %f", projection_value);
        if (projection_value > 0)
            atomicAdd(image_data + mi, value / (projection_value));
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
        const float *tof_t,
        const float tof_bin, const float tof_sigma2,
        const int gx, const int gy, const int gz,
        const float cx, const float cy, const float cz,
        const float sx, const float sy, const float sz,
        const int num_events,
        const float *image_data, float *projection_value)
{
    int grid[3] = {gx, gy, gz};
    float size[3] = {sx, sy, sz};
    float center[3] = {cx, cy, cz};
    TOF tof_info;
    tof_info.sigma2 = tof_sigma2;
    tof_info.binsize = tof_bin;
    tof_info.flag = tof_sigma2 < 20000? true: false;
    int step = blockDim.x * gridDim.x;
    // int jid = threadIdx.x;
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < (num_events + step); tid += step)
    {
        if (tid < num_events)
        {
            Ray ray;
            Block imgbox;
            // step1: create the ray and image block.
            CreateRay(x1[tid], y1[tid], z1[tid], x2[tid], y2[tid], z2[tid], ray);

            CreateBlock(grid, size, center, imgbox);
            // step2: judge if the ray pass through the image region.
            if (IsThroughImage(imgbox, ray))
            {
                // step3: cast the ray.
                RayCast raycast;
                SetupRayCast(imgbox, ray, raycast);
                float weight = 1 / ray.length / ray.length;
                float t_tof = ray.length * 0.5 - ray.min_t - tof_t[tid];

                // step4: raytracing the raycast and integrate the ray.
                RayTracing(tof_info, t_tof, weight, raycast, image_data, projection_value[tid]);
            }
            else
            {
                projection_value[tid] = 0.0;
            }
        }
    }
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
            const float *projection_value, float *image_data)
{
    int grid[3] = {gx, gy, gz};
    float size[3] = {sx, sy, sz};
    float center[3] = {cx, cy, cz};
    TOF tof_info;
    tof_info.sigma2 = tof_sigma2;
    tof_info.binsize = tof_bin;
    tof_info.flag = tof_sigma2 < 20000? true: false;
    int step = blockDim.x * gridDim.x;
    // int jid = threadIdx.x;
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < (num_events + step); tid += step)
    {
        if (tid < num_events)
        {
            Ray ray;
            Block imgbox;
            // step1: create the ray and image block.
            CreateRay(x1[tid], y1[tid], z1[tid], x2[tid], y2[tid], z2[tid], ray);
            CreateBlock(grid, size, center, imgbox);
            // step2: judge if the ray pass through the image region.
            if (IsThroughImage(imgbox, ray))
            {
                // step3: cast the ray.
                RayCast raycast;
                SetupRayCast(imgbox, ray, raycast);
                float weight = 1 / ray.length / ray.length;
                float t_tof = ray.length * 0.5 - ray.min_t - tof_t[tid];

                // step4: raytracing the raycast and integrate the ray.
                BackRayTracing(tof_info, t_tof, weight, raycast, projection_value[tid], image_data);
            }
            else
            {
                // projection_value[tid] = 0.0;
            }
        }
    }
}

/*
Function description:
This function do the paralell computing of lor backward mapping.
   Args:
 
   Returns:
*/
__global__ void
mapping(const float *x1, const float *y1, const float *z1,
        const float *x2, const float *y2, const float *z2,
        const int gx, const int gy, const int gz,
        const float cx, const float cy, const float cz,
        const float sx, const float sy, const float sz,
        const int num_events,
        const float *projection_value, float *image_data)
{
    int grid[3] = {gx, gy, gz};
    float size[3] = {sx, sy, sz};
    float center[3] = {cx, cy, cz};

    int step = blockDim.x * gridDim.x;
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < (num_events + step); tid += step)
    {
        if (tid < num_events)
        {
            Ray ray;
            Block imgbox;
            // step1: create the ray and image block.
            CreateRay(x1[tid], y1[tid], z1[tid], x2[tid], y2[tid], z2[tid], ray);
            float weight = 1 / (ray.length * ray.length);
            CreateBlock(grid, size, center, imgbox);
            // step2: judge if the ray pass through the image region.
            if (IsThroughImage(imgbox, ray))
            {
                // printf("the kernel was called!\n");
                // step3: cast the ray.
                RayCast raycast;
                SetupRayCast(imgbox, ray, raycast);
                // step4: raytracing the raycast and integrate the ray.
                Map(weight, raycast, projection_value[tid], image_data);
            }
            else
            {
                // projection_value[tid] = 0.0;
            }
        }
    }
}

/*
Function description:
   Args:
 
   Returns:
*/
void projection(const float *x1, const float *y1, const float *z1,
                const float *x2, const float *y2, const float *z2,
                const float *tof_t, float *projection_value,
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
    // std::cout << gx << " " << gy << " " << gz << std::endl;
    float cx = center_cpu[0], cy = center_cpu[1], cz = center_cpu[2]; // position of center
    float sx = size_cpu[0], sy = size_cpu[1], sz = size_cpu[2];

    project<<<GRIDDIM, BLOCKDIM>>>(x1, y1, z1,
                                   x2, y2, z2, tof_t,
                                   tof_bin, tof_sigma2,
                                   gx, gy, gz, cx, cy, cz, sx, sy, sz,
                                   num_events, image, projection_value);
}

/*
Function description:
   Args:
 
   Returns:
*/
void backprojection(const float *x1, const float *y1, const float *z1,
                    const float *x2, const float *y2, const float *z2,
                    const float *tof_t, const float *projection_value,
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
    // std::cout << gx << " " << gy << " " << gz << std::endl;
    float cx = center_cpu[0], cy = center_cpu[1], cz = center_cpu[2]; // position of center
    float sx = size_cpu[0], sy = size_cpu[1], sz = size_cpu[2];

    backproject<<<GRIDDIM, BLOCKDIM>>>(x1, y1, z1,
                                       x2, y2, z2,
                                       tof_t,
                                       tof_bin, tof_sigma2,
                                       gx, gy, gz,
                                       cx, cy, cz,
                                       sx, sy, sz,
                                       num_events,
                                       projection_value, image);
}

/*
Function description:
This function backproject the  
   Args:
 
   Returns:
*/
void maplors(const float *x1, const float *y1, const float *z1,
             const float *x2, const float *y2, const float *z2,
             const float *projection_value,
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
    // std::cout << gx << " " << gy << " " << gz << std::endl;
    float cx = center_cpu[0], cy = center_cpu[1], cz = center_cpu[2]; // position of center
    float sx = size_cpu[0], sy = size_cpu[1], sz = size_cpu[2];

    // debug
    std::cout << "grid: " << gx << ", " << gy << ", " << gz << std::endl;
    std::cout << "center: " << cx << ", " << cy << ", " << cz << std::endl;
    std::cout << "size: " << sx << ", " << sy << ", " << sz << std::endl;
    std::cout << "number of events: " << num_events << std::endl;
    mapping<<<GRIDDIM, BLOCKDIM>>>(x1, y1, z1, x2, y2, z2,
                                   gx, gy, gz, cx, cy, cz, sx, sy, sz,
                                   num_events, projection_value, image);
}

#endif