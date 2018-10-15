TF_CUSTOM_OP_IMPL_ROOT=~/Workspace/SRF/src/cpp/tf_custom_ops
TF_CUSTOM_OP_ROOT=~/tensorflow/tensorflow/core/user_ops
cp $TF_CUSTOM_OP_IMPL_ROOT/BUILD $TF_CUSTOM_OP_ROOT
cp $TF_CUSTOM_OP_IMPL_ROOT/*.cc $TF_CUSTOM_OP_ROOT
cp $TF_CUSTOM_OP_IMPL_ROOT/*.cu.cc $TF_CUSTOM_OP_ROOT

cd $TF_CUSTOM_OP_ROOT
bazel build --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --config opt //tensorflow/core/user_ops:tof_tor.so
bazel build --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --config opt //tensorflow/core/user_ops:tor.so
bazel build --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --config opt //tensorflow/core/user_ops:siddon.so
bazel build --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --config opt //tensorflow/core/user_ops:siddon2.so
bazel build --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --config opt //tensorflow/core/user_ops:siddon3.so

cd $TF_CUSTOM_OP_IMPL_ROOT