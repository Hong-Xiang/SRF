#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <cmath>
#include "cuda.h"
#include "cuda_runtime.h"
//#include "/home/chengaoyu/code/C++/BBSLMIRP_QT/PETSystem/petapplication.h"

#include <ctime>

using namespace tensorflow;
//using namespace BBSLMIRP;

REGISTER_OP("Projection")
    .Input("sino: float")
    .Input("image: float")
    .Input("grid: int32")
    .Input("center: float")
    .Input("size: float")
    .Input("block_grid: int32")
    .Input("block_center: float")
    .Input("block_size: float")
    .Output("result: float")
    .Attr("inner_radius: float")
    .Attr("outer_radius: float")
    .Attr("nb_rings: int")
    .Attr("nb_blocks_per_ring: int")
    .Attr("gap: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });

REGISTER_OP("Backprojection")
    .Input("image: float")
    .Input("sino: float")
    .Input("grid: int32")
    .Input("center: float")
    .Input("size: float")
    .Input("block_grid: int32")
    .Input("block_center: float")
    .Input("block_size: float")
    .Output("result: float")
    .Attr("inner_radius: float")
    .Attr("outer_radius: float")
    .Attr("nb_rings: int")
    .Attr("nb_blocks_per_ring: int")
    .Attr("gap: float")

    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
        //set the size of backpro_image the same as the input image.
        c->set_output(0, c->input(0));
        return Status::OK();
    });

REGISTER_OP("MapSino")
    .Input("image: float")
    .Input("sino: float")
    .Input("grid: int32")
    .Input("center: float")
    .Input("size: float")
    .Input("block_grid: int32")
    .Input("block_center: float")
    .Input("block_size: float")
    .Output("result: float")
    .Attr("inner_radius: float")
    .Attr("outer_radius: float")
    .Attr("nb_rings: int")
    .Attr("nb_blocks_per_ring: int")
    .Attr("gap: float")

    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
        //set the size of backpro_image the same as the input image.
        c->set_output(0, c->input(0));
        return Status::OK();
    });

void projection(float *result, const float *projection_value, const float *image,
                const int *grid, const float *center, const float *size,
                const int *block_grid, const float *block_center, const float *block_size,
                const float inner_radius, const float outer_radius, const int nb_rings,
                const int nb_blocks_per_ring, const float gap);

void backprojection(float *image, const float *projection,
                const int *grid, const float *center, const float *size,
                const int *block_grid, const float *block_center, const float *block_size,
                const float inner_radius, const float outer_radius, const int nb_rings,
                const int nb_blocks_per_ring, const float gap);

void mapsino(float *image, const float *projection_value, 
                const int *grid, const float *center, const float *size,
                const int *block_grid, const float *block_center, const float *block_size,
                const float inner_radius, const float outer_radius, const int nb_rings,
                const int nb_blocks_per_ring, const float gap);

class Projection : public OpKernel
{
  public:
    explicit Projection(OpKernelConstruction *context) : OpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("inner_radius", &inner_radius));
        OP_REQUIRES_OK(context, context->GetAttr("outer_radius", &outer_radius));
        OP_REQUIRES_OK(context, context->GetAttr("nb_rings", &nb_rings));
        OP_REQUIRES_OK(context, context->GetAttr("nb_blocks_per_ring", &nb_blocks_per_ring));
        OP_REQUIRES_OK(context, context->GetAttr("gap", &gap));
    }

    void Compute(OpKernelContext *context) override
    {
        // Grab the input tensor
        const Tensor &sino = context->input(0);
        const Tensor &image = context->input(1);
        const Tensor &grid = context->input(2);
        const Tensor &center = context->input(3);
        const Tensor &size = context->input(4);
        const Tensor &block_grid = context->input(5);
        const Tensor &block_center = context->input(6);
        const Tensor &block_size = context->input(7);
        // Create an output tensor
        Tensor *result = NULL;


        OP_REQUIRES_OK(context, context->allocate_output(0, sino.shape(),
                                                         &result));
        auto pv_flat = result->flat<float>();
        cudaMemset(pv_flat.data(), 0, sizeof(float) * pv_flat.size());

        auto grid_flat = grid.flat<int>();
        auto center_flat = center.flat<float>();
        auto size_flat = size.flat<float>();
        auto block_grid_flat = block_grid.flat<int>();
        auto block_center_flat = block_center.flat<float>();
        auto block_size_flat = block_size.flat<float>();
        auto image_flat = image.flat<float>();
        auto sino_flat = sino.flat<float>();

        projection(pv_flat.data(), sino_flat.data(), image_flat.data(),
                grid_flat.data(), center_flat.data(), size_flat.data(),
                block_grid_flat.data(), block_center_flat.data(), block_size_flat.data(),
                inner_radius, outer_radius, nb_rings, nb_blocks_per_ring, gap);
    }
    private:
        float inner_radius;
        float outer_radius;
        int nb_rings;
        int nb_blocks_per_ring;
        float gap;
};

class Backprojection : public OpKernel
{
  public:
    explicit Backprojection(OpKernelConstruction *context) : OpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("inner_radius", &inner_radius));
        OP_REQUIRES_OK(context, context->GetAttr("outer_radius", &outer_radius));
        OP_REQUIRES_OK(context, context->GetAttr("nb_rings", &nb_rings));
        OP_REQUIRES_OK(context, context->GetAttr("nb_blocks_per_ring", &nb_blocks_per_ring));
        OP_REQUIRES_OK(context, context->GetAttr("gap", &gap));
    }

    void Compute(OpKernelContext *context) override
    {
        // Grab the input tensor
        const Tensor &image = context->input(0);
        const Tensor &sino = context->input(1);
        const Tensor &grid = context->input(2);
        const Tensor &center = context->input(3);
        const Tensor &size = context->input(4);
        const Tensor &block_grid = context->input(5);
        const Tensor &block_center = context->input(6);
        const Tensor &block_size = context->input(7);
        // Create an output tensor
        Tensor *result = NULL;


        OP_REQUIRES_OK(context, context->allocate_output(0, image.shape(),
                                                         &result));
        auto bv_flat = result->flat<float>();
        cudaMemset(bv_flat.data(), 0, sizeof(float) * bv_flat.size());

        auto grid_flat = grid.flat<int>();
        auto center_flat = center.flat<float>();
        auto size_flat = size.flat<float>();
        auto block_grid_flat = block_grid.flat<int>();
        auto block_center_flat = block_center.flat<float>();
        auto block_size_flat = block_size.flat<float>();
        auto sino_flat = sino.flat<float>();

        backprojection(bv_flat.data(), sino_flat.data(),
                grid_flat.data(), center_flat.data(), size_flat.data(),
                block_grid_flat.data(), block_center_flat.data(), block_size_flat.data(),
                inner_radius, outer_radius, nb_rings, nb_blocks_per_ring, gap);
    }
    private:
        float inner_radius;
        float outer_radius;
        int nb_rings;
        int nb_blocks_per_ring;
        float gap;
};

class MapSino : public OpKernel
{
  public:
    explicit MapSino(OpKernelConstruction *context) : OpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("inner_radius", &inner_radius));
        OP_REQUIRES_OK(context, context->GetAttr("outer_radius", &outer_radius));
        OP_REQUIRES_OK(context, context->GetAttr("nb_rings", &nb_rings));
        OP_REQUIRES_OK(context, context->GetAttr("nb_blocks_per_ring", &nb_blocks_per_ring));
        OP_REQUIRES_OK(context, context->GetAttr("gap", &gap));
    }

    void Compute(OpKernelContext *context) override
    {
        // Grab the input tensor
        const Tensor &image = context->input(0);
        const Tensor &sino = context->input(1);
        const Tensor &grid = context->input(2);
        const Tensor &center = context->input(3);
        const Tensor &size = context->input(4);
        const Tensor &block_grid = context->input(5);
        const Tensor &block_center = context->input(6);
        const Tensor &block_size = context->input(7);
        // Create an output tensor
        Tensor *result = NULL;


        OP_REQUIRES_OK(context, context->allocate_output(0, image.shape(),
                                                         &result));
        auto bv_flat = result->flat<float>();
        cudaMemset(bv_flat.data(), 0, sizeof(float) * bv_flat.size());

        auto grid_flat = grid.flat<int>();
        auto center_flat = center.flat<float>();
        auto size_flat = size.flat<float>();
        auto block_grid_flat = block_grid.flat<int>();
        auto block_center_flat = block_center.flat<float>();
        auto block_size_flat = block_size.flat<float>();
        auto sino_flat = sino.flat<float>();

        mapsino(bv_flat.data(), sino_flat.data(),
                grid_flat.data(), center_flat.data(), size_flat.data(),
                block_grid_flat.data(), block_center_flat.data(), block_size_flat.data(),
                inner_radius, outer_radius, nb_rings, nb_blocks_per_ring, gap);
    }
    private:
    float inner_radius;
    float outer_radius;
    int nb_rings;
    int nb_blocks_per_ring;
    float gap;
};

#define REGISTER_GPU_KERNEL(name, op) \
    REGISTER_KERNEL_BUILDER(          \
        Name(name).Device(DEVICE_GPU), op)

REGISTER_GPU_KERNEL("Projection", Projection);
REGISTER_GPU_KERNEL("Backprojection", Backprojection);
REGISTER_GPU_KERNEL("MapSino", MapSino);

#undef REGISTER_GPU_KERNEL