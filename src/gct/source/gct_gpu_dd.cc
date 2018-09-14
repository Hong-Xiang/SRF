#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <cmath>
#include <algorithm>
#include "cuda.h"
#include "cuda_runtime.h"
#include <vector>
using namespace tensorflow;

#include "../include/gct_Image.h"
#include "../include/gct_Projection.h"
#include "../include/gct_gpu_dd.h"
// using namespace gct;

REGISTER_OP("Projection")
    .Input("img:float")
    .Output("result: float")
    .Attr("imgGrid: list(int)")// img parameters
    .Attr("imgSize: list(float)")
    .Attr("imgCenter: list(float)")
    .Attr("projGrid: list(int)")// projection parameters
    .Attr("projSize: list(float)")
    .Attr("projCenter: list(float)")
    .Attr("SID: float")
    .Attr("SAD: float")
    .Attr("angle: float");
    // .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
    //     c->set_output(0, c->Matrix(c->Dim(c->input(1), 1), 1));
    //     return Status::OK();
    // });

class Projection : public OpKernel
{
	public:
		explicit Projection(OpKernelConstruction *context) : OpKernel(context)
		{
			OP_REQUIRES_OK(context, context->GetAttr("imgGrid", &imgGrid));
			OP_REQUIRES_OK(context, context->GetAttr("imgSize", &imgSize));
			OP_REQUIRES_OK(context, context->GetAttr("imgCenter", &imgCenter));
			OP_REQUIRES_OK(context, context->GetAttr("projGrid", &projGrid));
			OP_REQUIRES_OK(context, context->GetAttr("projSize", &projSize));
			OP_REQUIRES_OK(context, context->GetAttr("projCenter", &projCenter));
			OP_REQUIRES_OK(context, context->GetAttr("SAD", &SAD));
			OP_REQUIRES_OK(context, context->GetAttr("SID", &SID));
			OP_REQUIRES_OK(context, context->GetAttr("angle", &angle));
		}

		
		void Compute(OpKernelContext *context) override
		{
			const Tensor &img = context->input(0);

			// TODO: Better way of constructions?
			TensorShape out_shape({projGrid[0], projGrid[1]});
			// out_shape.AddDim(img.shape().dim_size(0));
			Tensor *result = nullptr;
			OP_REQUIRES_OK(context, context->allocate_output(0, out_shape,
															&result));
			auto result_flat = result->flat<float>();
			cudaMemset(result_flat.data(), 0, sizeof(float) * result_flat.size());
			ProjectionKernelLauncher(img.flat<float>().data(),								
									ImgInfo{imgGrid, imgSize, imgCenter},
									ProjInfo{projGrid, projSize, projCenter, SID, SAD, angle},
									result_flat.data());
		}							

  	private:
		// gct::ImgInfo <3> );
		// gct::ProjInfo <2> projInfo();
		std::vector <int> imgGrid;
		std::vector <float> imgSize;
		std::vector <float> imgCenter;
		std::vector <int> projGrid;
		std::vector <float> projSize;
		std::vector <float> projCenter;
		float SID;
		float SAD;
		float angle;
};


REGISTER_OP("Backprojection")
    .Input("proj:float")
    .Output("result: float")
    .Attr("imgGrid: list(int)")// image parameters
    .Attr("imgSize: list(float)")
    .Attr("imgCenter: list(float)")
    .Attr("projGrid: list(int)")// projection parameters
    .Attr("projSize: list(float)")
    .Attr("projCenter: list(float)")
    .Attr("SID: float")
    .Attr("SAD: float")
    .Attr("angle: float");
    // .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
    //     c->set_output(0, c->Matrix(c->Dim(c->input(1), 1), 1));
    //     return Status::OK();
    // });

class Backprojection : public OpKernel
{
	public:
		explicit Backprojection(OpKernelConstruction *context) : OpKernel(context)
		{
			OP_REQUIRES_OK(context, context->GetAttr("imgGrid", &imgGrid));
			OP_REQUIRES_OK(context, context->GetAttr("imgSize", &imgSize));
			OP_REQUIRES_OK(context, context->GetAttr("imgCenter", &imgCenter));
			OP_REQUIRES_OK(context, context->GetAttr("projGrid", &projGrid));
			OP_REQUIRES_OK(context, context->GetAttr("projSize", &projSize));
			OP_REQUIRES_OK(context, context->GetAttr("projCenter", &projCenter));
			OP_REQUIRES_OK(context, context->GetAttr("SAD", &SAD));
			OP_REQUIRES_OK(context, context->GetAttr("SID", &SID));
			OP_REQUIRES_OK(context, context->GetAttr("angle", &angle));
		}

		
		void Compute(OpKernelContext *context) override
		{
			const Tensor &proj = context->input(0);

			// TODO: Better way of constructions?
			TensorShape out_shape({imgGrid[0], imgGrid[1], imgGrid[2]});
			// out_shape.AddDim(img.shape().dim_size(0));
			Tensor *result = nullptr;
			OP_REQUIRES_OK(context, context->allocate_output(0, out_shape,
															&result));
			auto result_flat = result->flat<float>();
			cudaMemset(result_flat.data(), 0, sizeof(float) * result_flat.size());
			BackprojectionKernelLauncher(proj.flat<float>().data(),								
									ImgInfo{imgGrid, imgSize, imgCenter},
									ProjInfo{projGrid, projSize, projCenter, SID, SAD, angle},
									result_flat.data());
		}							

  	private:
		// gct::ImgInfo <3> );
		// gct::ProjInfo <2> projInfo();
		std::vector <int> imgGrid;
		std::vector <float> imgSize;
		std::vector <float> imgCenter;
		std::vector <int> projGrid;
		std::vector <float> projSize;
		std::vector <float> projCenter;
		float SID;
		float SAD;
		float angle;
};



#define REGISTER_GPU_KERNEL(name, op) \
    REGISTER_KERNEL_BUILDER(          \
        Name(name).Device(DEVICE_GPU), op)

REGISTER_GPU_KERNEL("Projection", Projection);

#undef REGISTER_GPU_KERNEL