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
  block_wmma_async,
};

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                      KernelType kernel_type);

}
