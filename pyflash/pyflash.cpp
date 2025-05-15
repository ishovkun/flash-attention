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

static torch::Tensor forward_scalar2d_row_tile(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
  return flash::forward(Q, K, V, flash::KernelType::scalar2D_row_tile);
}

static torch::Tensor forward_warp_wmma(torch::Tensor Q, torch::Tensor K,
                                            torch::Tensor V) {
  return flash::forward(Q, K, V, flash::KernelType::warp_wmma);
}

static torch::Tensor forward_block_wmma(torch::Tensor Q, torch::Tensor K,
                                             torch::Tensor V) {
  return flash::forward(Q, K, V, flash::KernelType::block_wmma);
}

static torch::Tensor forward_wmma_row_block(torch::Tensor Q, torch::Tensor K,
                                             torch::Tensor V) {
  return flash::forward(Q, K, V, flash::KernelType::wmma_row_block);
}

static torch::Tensor forward_block_wmma_async(torch::Tensor Q, torch::Tensor K,
                                              torch::Tensor V) {
  return flash::forward(Q, K, V, flash::KernelType::block_wmma);
}

static torch::Tensor forward_mma(torch::Tensor Q, torch::Tensor K,
                                       torch::Tensor V) {
  return flash::forward(Q, K, V, flash::KernelType::mma);
}

static torch::Tensor forward_mma_swizzle(torch::Tensor Q, torch::Tensor K,
                                       torch::Tensor V) {
  return flash::forward(Q, K, V, flash::KernelType::mma_swizzle);
}


TORCH_LIBRARY(pyflash, m) {
  m.def("naive(Tensor Q, Tensor K, Tensor V) -> Tensor",
        torch::wrap_pybind_function(forward_naive));
  m.def("scalar2d(Tensor Q, Tensor K, Tensor V) -> Tensor",
        torch::wrap_pybind_function(forward_scalar2d));
  m.def("scalar2d_row_tile(Tensor Q, Tensor K, Tensor V) -> Tensor",
        torch::wrap_pybind_function(forward_scalar2d_row_tile));
  m.def("warp_wmma(Tensor Q, Tensor K, Tensor V) -> Tensor",
        torch::wrap_pybind_function(forward_warp_wmma));
  m.def("block_wmma(Tensor Q, Tensor K, Tensor V) -> Tensor",
        torch::wrap_pybind_function(forward_block_wmma));
  m.def("block_wmma_row_block(Tensor Q, Tensor K, Tensor V) -> Tensor",
        torch::wrap_pybind_function(forward_wmma_row_block));
  m.def("block_wmma_async(Tensor Q, Tensor K, Tensor V) -> Tensor",
        torch::wrap_pybind_function(forward_block_wmma_async));
  m.def("mma(Tensor Q, Tensor K, Tensor V) -> Tensor",
        torch::wrap_pybind_function(forward_mma));
  m.def("mma_swizzle(Tensor Q, Tensor K, Tensor V) -> Tensor",
        torch::wrap_pybind_function(forward_mma_swizzle));
}
