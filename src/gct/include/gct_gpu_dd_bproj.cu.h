#ifndef _GCT_GPU_DD_BPROJ_CU_H
#define _GCT_GPU_DD_BPROJ_CU_H
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "cuda.h"
#include "cuda_runtime.h"
#include "gct_Image.h"
#include "gct_Projection.h"

// using namespace gct;
#ifndef BLOCKSIZE_X
#define BLOCKSIZE_X 32
#define BLOCKSIZE_Y 32
#define BLOCKSIZE_Z 4
#endif // BLOCKSIZE_X
#ifndef MAX
#define MAX(a,b) (((a) > (b)) ? a : b)
#define MIN(a,b) (((a) < (b)) ? a : b)
#endif // MAX

#ifndef MAX4
#define MAX4(a, b, c, d) MAX(MAX(a, b), MAX(c, d))
#define MIN4(a, b, c, d) MIN(MIN(a, b), MIN(c, d))
#endif //MAX4

#ifndef MAX6
#define MAX6(a, b, c, d, e, f) MAX(MAX(MAX(a, b), MAX(c, d)), MAX(e, f))
#define MIN6(a, b, c, d, e, f) MIN(MIN(MIN(a, b), MIN(c, d)), MIN(e, f))
#endif //MAX6

#ifndef ABS
#define ABS(x) ((x) > 0 ? x : -(x))
#endif // ABS

#ifndef PI
#define PI 3.141592653589793f
#endif //PI
void BackprojectionKernelLauncher(const float *proj,
							  const ImgInfo imgInfo,
							  const ProjInfo projInfo,
							  float *result);

__global__ void BackprojectionKernel(cudaTextureObject_t tex_proj, float angle, float SO, float SD, float da, int na,
	float ai, float db, int nb, float bi, int nx, int ny, int nz, float *result);

#endif //  _GCT_GPU_DD_BPROJ_CU_H