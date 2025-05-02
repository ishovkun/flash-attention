#pragma once
#include <torch/torch.h>

namespace flash {

enum class KernelType {
  naive1D,
  scalar2D,
  warp_wmma_sync,
  block_wmma_sync,
  block_wmma_async,
};

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                      KernelType kernel_type);

}
