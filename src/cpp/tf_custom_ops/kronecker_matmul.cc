// This function provides the method to calculate the kronecker product
// of seperated psf kernels (kernel_xy and kernel_z) and matrix multiplication
// with the 3D image.

// This method is required because the huge PSF kernel can not be
// loaded into the limited memory(GPU or CPU). Thus the

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <cmath>
#include "cuda.h"
#include "cuda_runtime.h"

using namespace tensorflow;

REGISTER_OP("KroneckerMatmulGPU")
    .Input("x: float")
    .Input("x_shape: int32")
    .Input("kernel_xy_indices: int32")
    .Input("kernel_xy_values: float")
    .Input("kernel_xy_shape: int32")
    .Input("kernel_z_indices: int32")
    .Input("kernel_z_values: float")
    .Input("kernel_z_shape: int32")

    .Output("result: float")

    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
        c->set_output(0, c->input(0)));
        return Status::OK();
    });

void kronecker_matmul(const int *kernel_xy_indices, const float *kernel_xy_values, const int *kernel_xy_shape,
                      const int *kernel_z_indices, const float *kernel_z_values, const int kernel_z_shape,
                      const float *x, const int *x_shape, float *reuslt);

class KroneckerMatmul : public OpKernel
{
  public:
    explicit KroneckerMatmul(OpKernelConstruction *context) : OpKernel(context)
    {
    }
    void Compute(OpKernelContext *context) override
    {
        const Tensor &image = context->input(0);
    }
};

#define REGISTER_GPU_KERNEL(name, op) \
    REGISTER_KERNEL_BUILDER(          \
        Name(name).Device(DEVICE_GPU), op)

REGISTER_GPU_KERNEL("KroneckerMatmulGPU", KroneckerMatmul);