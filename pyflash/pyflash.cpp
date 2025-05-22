#include "flash/launch.hpp"
#include <torch/extension.h>

template <flash::KernelType kernel_type>
static torch::Tensor forward_kernel_wrapper(torch::Tensor Q, torch::Tensor K,
                                            torch::Tensor V) {
  return flash::forward(Q, K, V, kernel_type);
}

TORCH_LIBRARY(pyflash, m) {
  m.def("naive(Tensor Q, Tensor K, Tensor V) -> Tensor",
        torch::wrap_pybind_function(forward_kernel_wrapper<flash::KernelType::naive1D>));
  m.def("scalar2d(Tensor Q, Tensor K, Tensor V) -> Tensor",
        torch::wrap_pybind_function(forward_kernel_wrapper<flash::KernelType::scalar2D>));
  m.def("scalar2d_row_tile(Tensor Q, Tensor K, Tensor V) -> Tensor",
        torch::wrap_pybind_function(forward_kernel_wrapper<flash::KernelType::scalar2D_row_tile>));
  m.def("warp_wmma(Tensor Q, Tensor K, Tensor V) -> Tensor",
        torch::wrap_pybind_function(forward_kernel_wrapper<flash::KernelType::warp_wmma>));
  m.def("block_wmma(Tensor Q, Tensor K, Tensor V) -> Tensor",
        torch::wrap_pybind_function(forward_kernel_wrapper<flash::KernelType::block_wmma>));
  m.def("block_wmma_row_block(Tensor Q, Tensor K, Tensor V) -> Tensor",
        torch::wrap_pybind_function(forward_kernel_wrapper<flash::KernelType::wmma_row_block>));
  m.def("block_wmma_async(Tensor Q, Tensor K, Tensor V) -> Tensor",
        torch::wrap_pybind_function(forward_kernel_wrapper<flash::KernelType::block_wmma_async>));
  m.def("mma(Tensor Q, Tensor K, Tensor V) -> Tensor",
        torch::wrap_pybind_function(forward_kernel_wrapper<flash::KernelType::mma>));
  m.def("mma_swizzle(Tensor Q, Tensor K, Tensor V) -> Tensor",
        torch::wrap_pybind_function(forward_kernel_wrapper<flash::KernelType::mma_swizzle>));
  m.def("mma_qreg(Tensor Q, Tensor K, Tensor V) -> Tensor",
        torch::wrap_pybind_function(forward_kernel_wrapper<flash::KernelType::mma_qreg>));
  m.def("mma_qreg_vld(Tensor Q, Tensor K, Tensor V) -> Tensor",
        torch::wrap_pybind_function(forward_kernel_wrapper<flash::KernelType::mma_qreg_f32x4load>));
  m.def("mma_qreg_async(Tensor Q, Tensor K, Tensor V) -> Tensor",
        torch::wrap_pybind_function(forward_kernel_wrapper<flash::KernelType::mma_qreg_async>));
}
