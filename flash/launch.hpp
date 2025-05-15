#pragma once
#include <torch/torch.h>
#include <stdexcept>

namespace flash {

enum class KernelType {
  naive1D,
  scalar2D,
  scalar2D_row_tile,
  warp_wmma_sync,
  block_wmma_sync,
  wmma_sync_row_block,
  block_wmma_async,
  mma_sync,
  mma_sync_swizzle,
  mma_sync_qreg,
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
    case warp_wmma_sync: return "warp_wmma_sync";
    case block_wmma_sync: return "block_wmma_sync";
    case wmma_sync_row_block: return "wmma_sync_row_block";
    case block_wmma_async: return "block_wmma_async";
    case mma_sync: return "mma_sync";
    case mma_sync_swizzle: return "mma_sync_swizzle";
    default: throw std::invalid_argument("wrong kernel type");
  }
}
