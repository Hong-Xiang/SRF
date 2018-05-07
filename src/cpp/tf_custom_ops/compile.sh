TF_CUSTOME_OP_IMPL_ROOT=~/Workspace/SRF/src/cpp/tf_custom_ops

rsync $TF_CUSTOME_OP_IMPL_ROOT/BUILD .
rsync $TF_CUSTOME_OP_IMPL_ROOT/*.cc .
rsync $TF_CUSTOME_OP_IMPL_ROOT/*.cu.cc .

bazel build --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --config opt //tensorflow/core/user_ops:pet_gpu.so
bazel build --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --config opt //tensorflow/core/user_ops:tof_tor.so