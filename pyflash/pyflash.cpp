#include <torch/extension.h>
#include "flash/launch.hpp"

static torch::Tensor forward_naive(torch::Tensor Q, torch::Tensor K,
                                   torch::Tensor V) {
  return flash::forward(Q, K, V, flash::KernelType::naive1D);
}

static torch::Tensor forward_scalar2d(torch::Tensor Q, torch::Tensor K,
                                      torch::Tensor V) {
  return flash::forward(Q, K, V, flash::KernelType::scalar2D);
}

TORCH_LIBRARY(pyflash, m) {
  m.def("naive(Tensor Q, Tensor K, Tensor V) -> Tensor",
        torch::wrap_pybind_function(forward_naive));
  m.def("scalar2d(Tensor Q, Tensor K, Tensor V) -> Tensor",
        torch::wrap_pybind_function(forward_scalar2d));
}
