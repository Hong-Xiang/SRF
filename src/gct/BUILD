load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    name = "gct_gpu_dd.so",
    srcs = [
        # glob("include/*.h"),
        "include/gct_gpu_dd.h",
        "include/gct_gpu_dd_proj.cu.h", 
        "include/gct_gpu_dd_bproj.cu.h", 
        "include/gct_Image.h",
        "include/gct_Projection.h",
        "source/gct_gpu_dd.cc"
        ],
    gpu_srcs = [
        "source/gct_gpu_dd_proj.cu.cc",
        "source/gct_gpu_dd_bproj.cu.cc"
        ],
)
