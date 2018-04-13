
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    name = "pet_gpu.so",
    srcs = ["pet_gpu.cc"],
    gpu_srcs = ["pet_gpu.cu.cc"],
)

tf_custom_op_library(
    name = "tof_tor.so",
    srcs = ["tof_tor.cc"],
    gpu_srcs = ["tof_tor.cu.cc"]
)

tf_custom_op_library(
    name = "siddon_gpu.so",
    srcs = ["siddon_gpu.cc"],
    gpu_srcs = ["siddon_gpu.cu.cc"]
)

