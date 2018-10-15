load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

# tf_custom_op_library(
#     name = "pet_gpu.so",
#     srcs = ["pet_gpu.cc"],
#     gpu_srcs = ["pet_gpu.cu.cc"],
# )

tf_custom_op_library(
    name = "tor.so",
    srcs = ["tor.cc"],
    gpu_srcs = ["tor.cu.cc"],
)

tf_custom_op_library(
    name = "tof_tor.so",
    srcs = ["tof_tor.cc"],
    gpu_srcs = ["tof_tor.cu.cc"]
)

tf_custom_op_library(
    name = "tof_new.so",
    srcs = ["tof_new.cc"],
    gpu_srcs = ["tof_new.cu.cc"]
)


tf_custom_op_library(
    name = "siddon_gpu.so",
    srcs = ["siddon_gpu.cc"],
    gpu_srcs = ["siddon_gpu.cu.cc"]
)

tf_custom_op_library(
    name = "siddon.so",
    srcs = ["siddon.cc"],
    gpu_srcs = ["siddon.cu.cc"],
)


tf_custom_op_library(
    name = "siddon2.so",
    srcs = ["siddon.cc"],
    gpu_srcs = ["siddon2.cu.cc"],
)


tf_custom_op_library(
    name = "siddon3.so",
    srcs = ["siddon.cc"],
    gpu_srcs = ["siddon_crystal.cu.cc"],
)
