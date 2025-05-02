#include "flash/launch.hpp"
#include <torch/extension.h>

static torch::Tensor forward_naive(torch::Tensor Q, torch::Tensor K,
                                   torch::Tensor V) {
  return flash::forward(Q, K, V, flash::KernelType::naive1D);
}

static torch::Tensor forward_scalar2d(torch::Tensor Q, torch::Tensor K,
                                      torch::Tensor V) {
  return flash::forward(Q, K, V, flash::KernelType::scalar2D);
}

static torch::Tensor forward_warp_wmma_sync(torch::Tensor Q, torch::Tensor K,
                                       torch::Tensor V) {
  return flash::forward(Q, K, V, flash::KernelType::warp_wmma_sync);
}

static torch::Tensor forward_block_wmma_sync(torch::Tensor Q, torch::Tensor K,
                                             torch::Tensor V) {
  return flash::forward(Q, K, V, flash::KernelType::block_wmma_sync);
}

static torch::Tensor forward_block_wmma_async(torch::Tensor Q, torch::Tensor K,
                                              torch::Tensor V) {
  return flash::forward(Q, K, V, flash::KernelType::block_wmma_sync);
}

TORCH_LIBRARY(pyflash, m) {
  m.def("naive(Tensor Q, Tensor K, Tensor V) -> Tensor",
        torch::wrap_pybind_function(forward_naive));
  m.def("scalar2d(Tensor Q, Tensor K, Tensor V) -> Tensor",
        torch::wrap_pybind_function(forward_scalar2d));
  m.def("warp_wmma_sync(Tensor Q, Tensor K, Tensor V) -> Tensor",
        torch::wrap_pybind_function(forward_warp_wmma_sync));
  m.def("block_wmma_sync(Tensor Q, Tensor K, Tensor V) -> Tensor",
        torch::wrap_pybind_function(forward_block_wmma_sync));
  m.def("block_wmma_async(Tensor Q, Tensor K, Tensor V) -> Tensor",
        torch::wrap_pybind_function(forward_block_wmma_async));
}
