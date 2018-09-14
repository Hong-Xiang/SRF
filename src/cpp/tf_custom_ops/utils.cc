#include <array>
#include "tensorflow/core/framework/op.h"
using namespace tensorflow;

std::vector<const Tensor &> TransposeAndFlatten(const Tensor &t, int nb_columns)
{
    auto result = new std::vector<const Tensor &>(nb_columns);
    for (int i = 0; i < nb_columns; ++i)
        result[i] = t.Slice(i, i + 1).unaligned_flat<float>();
    return result
}