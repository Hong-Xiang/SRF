TF_CUSTOME_OP_IMPL_ROOT=~/Workspace/SRF/src/gct

cp $TF_CUSTOME_OP_IMPL_ROOT/BUILD .
# cp $TF_CUSTOME_OP_IMPL_ROOT/*.cc .
# cp $TF_CUSTOME_OP_IMPL_ROOT/*.cu.cc .
cp -R $TF_CUSTOME_OP_IMPL_ROOT/include .
cp -R $TF_CUSTOME_OP_IMPL_ROOT/source .

bazel build --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --config opt //tensorflow/core/user_ops:gct_gpu_dd.so