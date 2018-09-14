#ifndef GCT_GPU_DD_H
#define GCT_GPU_DD_H
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <cmath>
#include <algorithm>
#include "cuda.h"
#include "cuda_runtime.h"
#include <vector>
using namespace tensorflow;

#include "gct_Image.h"
#include "gct_Projection.h"
#include "gct_gpu_dd_proj.cu.h"
#include "gct_gpu_dd_bproj.cu.h"

#endif // GCT_GPU_DD_H
