#pragma once
#include <torch/torch.h>
#include <stdexcept>

namespace flash {

enum class KernelType {
  naive1D,
  scalar2D,
  scalar2D_row_tile,
  warp_wmma,
  block_wmma,
  wmma_row_block,
  block_wmma_async,
  mma,
  mma_swizzle,
  mma_qreg,
};


torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                      KernelType kernel_type);

}

inline std::string to_string(flash::KernelType type)
{
  using enum flash::KernelType;
  switch (type) {
    case naive1D: return "naive1D";
    case scalar2D: return "scalar2D";
    case scalar2D_row_tile: return "scalar2D_row_tile";
    case warp_wmma: return "warp_wmma";
    case block_wmma: return "block_wmma";
    case wmma_row_block: return "wmma_row_block";
    case block_wmma_async: return "block_wmma_async";
    case mma: return "mma";
    case mma_swizzle: return "mma_swizzle";
    case mma_qreg: return "mma_qreg";
    default: throw std::invalid_argument("wrong kernel type");
  }
}
