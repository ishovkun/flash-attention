#include "KernelParameters.hpp"
#include "common.hpp"
#include "launch.hpp"
#include "wmma.hpp"
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace flash {

__global__ void naive_forward_kernel(const float *Q, const float *K,
                                     const float *V, const int N, const int d,
                                     const int Tc, const int Tr, const int Bc,
                                     const int Br, const float softmax_scale,
                                     float *l, float *m, float *O);

__global__ void forward_kernel_2d(float const *__restrict__ Q,
                                  float const *__restrict__ K,
                                  float const *__restrict__ V, int N, int d,
                                  int Bc, int Br, float softmax_scale,
                                  float *__restrict__ l, float *__restrict__ m,
                                  float *__restrict__ O);

__global__ void
forward_kernel_2d_row_tile(float const *__restrict__ Q,
                           float const *__restrict__ K,
                           float const *__restrict__ V, int N, int d, int Bc,
                           int Br, float softmax_scale, float *__restrict__ l,
                           float *__restrict__ m, float *__restrict__ O);

__global__ void warp_wmma_sync(const float *Q, const float *K, const float *V,
                               const int N, const int d, const int Tc,
                               const int Tr, const int Bc, const int Br,
                               const float softmax_scale, float *l, float *m,
                               float *O);

__global__ void block_wmma_sync(float const *__restrict__ Q,
                                float const *__restrict__ K,
                                float const *__restrict__ V, int N, int d,
                                int Bc, int Br, float softmax_scale,
                                float *__restrict__ l, float *__restrict__ m,
                                float *__restrict__ O);

__global__ void wmma_sync_rowblock(float const *__restrict__ Q,
                                   float const *__restrict__ K,
                                   float const *__restrict__ V, int N, int d,
                                   int Bc, int Br, float softmax_scale,
                                   float *__restrict__ l, float *__restrict__ m,
                                   float *__restrict__ O);

__global__ void
forward_kernel_mma_sync(float const *__restrict__ Q,
                        float const *__restrict__ K,
                        float const *__restrict__ V, int N, int d, int Bc,
                        int Br, float softmax_scale, float *__restrict__ l,
                        float *__restrict__ m, float *__restrict__ O);

__global__ void block_wmma_async(float const *__restrict__ Q,
                                 float const *__restrict__ K,
                                 float const *__restrict__ V, int N, int d,
                                 int Bc, int Br, float softmax_scale,
                                 float *__restrict__ l, float *__restrict__ m,
                                 float *__restrict__ O);

// #define INVALID_ARGUMENT(msg)                                                  \
//   {                                                                            \
//     std::ostringstream err;                                                    \
//     err << __FILE__ << "(" << __LINE__ << ") "                                 \
//         << "error : " << msg;                                                  \
//     throw std::invalid_argument(err.str());                                    \
//   }

// static int maxTileSizeForDeviceSharedMemory(int head_dim) {
//   /*
//    * We compute the tile size assuming the maximum use of shared memory.
//    * This is done by solving a quadratic equation:
//    * sh_size = 3*Bc*d + (Bc*Br) -> solve for Bc
//    * Assume Bc = Br = x
//    * x = (sqrt(9*d*d + 4*m) - 3*d)/2
//    */
//   int max_sram_size;
//   cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock,
//   0); int const c = max_sram_size / sizeof(float); int d = head_dim; const
//   int tileSize = floor((sqrt(9. * d * d + 4. * c) - 3. * d) / 2.); return
//   tileSize;
// }

// static int selectTileSize(int d, KernelType kernelType) {
//   using namespace flash::constants;
//   switch (kernelType) {
//   case KernelType::naive1D:
//     return std::min(maxTileSizeForDeviceSharedMemory(d), maxBlockSize);
//   case KernelType::scalar2D:
//     return std::min(maxTileSizeForDeviceSharedMemory(d),
//                     maxBlockSize / warpSize);
//   case KernelType::warp_wmma_sync:
//     return constants::WMMA_M;
//   case KernelType::block_wmma_sync:
//     return std::min(maxTileSizeForDeviceSharedMemory(d),
//                     maxBlockSize / warpSize);
//   case KernelType::block_wmma_async:
//     return std::min(maxTileSizeForDeviceSharedMemory(d),
//                     maxBlockSize / warpSize);
//   default: { INVALID_ARGUMENT("Unsupported kernel type"); }
//   }
// }

// static dim3 selectBlockDim(int tileSize, KernelType kernelType) {
//   switch (kernelType) {
//   case KernelType::naive1D:
//     return tileSize;
//   case KernelType::scalar2D:
//     return dim3(tileSize, constants::warpSize);
//   case KernelType::warp_wmma_sync:
//     return constants::warpSize;
//   case KernelType::block_wmma_sync:
//     return dim3(tileSize, constants::warpSize);
//   case KernelType::block_wmma_async:
//     return dim3(tileSize, constants::warpSize);
//   default: { INVALID_ARGUMENT("Unsupported kernel type"); }
//   }
// }

static void checkRequestedSharedMemory(int requested_shared_memory) {
  int max_sram_size;
  cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
  // std::cout << "Requested shared memory: " << requested_shared_memory <<
  // std::endl; std::cout << "Maximum Shared memory: " << max_sram_size <<
  // std::endl;
  if (requested_shared_memory > max_sram_size) {
    std::cerr << "Requested shared memory " << requested_shared_memory
              << " exceeds maximum allowed (" << max_sram_size << ")"
              << std::endl;
    throw std::runtime_error("Requested shared memory exceeds maximum allowed");
  }
}

#define gpuErrchk(ans)                                                         \
  {                                                                            \
    gpuAssert((ans), __FILE__, __LINE__);                                      \
  }

static inline void gpuAssert(cudaError_t code, const char *file, int line,
                             bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    std::cout << "GPU assert failed" << std::endl;
    if (abort)
      exit(code);
  }
}

// static inline int selectPadding(int d, KernelType kernelType) {
//   if (kernelType == KernelType::warp_wmma_sync ||
//       kernelType == KernelType::block_wmma_sync ||
//       kernelType == KernelType::block_wmma_async)
//     return common::nextMultiple(d, constants::WMMA_N);
//   else
//     return d;
// }

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                      KernelType kernelType) {
  const int B = Q.size(0);  // batch size
  const int nh = Q.size(1); // number of heads
  const int N = Q.size(2);  // sequence length
  const int d = Q.size(3);  // head dimension

  // auto const d_pad = selectPadding(d, kernelType);
  auto kernelParams = FlashKernelParametersFactory::create(B, nh, N, d, kernelType);

  // auto const tileSize = selectTileSize(d_pad, kernelType);
  auto const tileSize = kernelParams->tileSize();

  auto const Br = tileSize.y;
  auto const Bc = tileSize.x;

  const float softmax_scale = 1.0 / sqrt(d);

  // Initialize O, l, m to HBM
  auto O = torch::zeros_like(Q);

  float *l, *m;
  cudaMalloc(&l, B * nh * N * sizeof(float));
  cudaMalloc(&m, B * nh * N * sizeof(float));

  // Calculate SRAM size needed per block
  auto const sramSize = kernelParams->sramSize();
  checkRequestedSharedMemory(sramSize);

  auto const gridDim = kernelParams->gridDim();
  auto const blockDim = kernelParams->blockDim();

  // std::cout << "Block dim: " << blockDim.x << ", " << blockDim.y << std::endl;
  // std::cout << "Grid dim: " << gridDim.x << ", " << gridDim.y << ", "
  //           << gridDim.z << std::endl;

  // number of tiles in K/V and Q, respectively
  auto const Tc = common::ceil_div(N, Bc);
  auto const Tr = common::ceil_div(N, Br);

  // launch kernel
  switch (kernelType) {
  case KernelType::naive1D: {
    naive_forward_kernel<<<gridDim, blockDim, sramSize>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), N, d, Tc,
        Tr, Bc, Br, softmax_scale, l, m, O.data_ptr<float>());
    break;
  }
  case KernelType::scalar2D:
    forward_kernel_2d<<<gridDim, blockDim, sramSize>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), N, d, Bc,
        Br, softmax_scale, l, m, O.data_ptr<float>());
    break;
  case KernelType::scalar2D_row_tile:
    forward_kernel_2d_row_tile<<<gridDim, blockDim, sramSize>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), N, d, Bc,
        Br, softmax_scale, l, m, O.data_ptr<float>());
    break;
  case KernelType::warp_wmma_sync: {
    warp_wmma_sync<<<gridDim, blockDim, sramSize>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), N, d, Tc,
        Tr, Bc, Br, softmax_scale, l, m, O.data_ptr<float>());
    break;
  }
  case KernelType::block_wmma_sync:
    block_wmma_sync<<<gridDim, blockDim, sramSize>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), N, d, Bc,
        Br, softmax_scale, l, m, O.data_ptr<float>());
    break;
  case KernelType::wmma_sync_row_block:
    wmma_sync_rowblock<<<gridDim, blockDim, sramSize>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), N, d, Bc,
        Br, softmax_scale, l, m, O.data_ptr<float>());
    break;
  case KernelType::block_wmma_async:
    block_wmma_async<<<gridDim, blockDim, sramSize>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), N, d, Bc,
        Br, softmax_scale, l, m, O.data_ptr<float>());
    break;
  case KernelType::mma_sync:
    forward_kernel_mma_sync<<<gridDim, blockDim, sramSize>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), N, d, Bc,
        Br, softmax_scale, l, m, O.data_ptr<float>());
    break;
  default:
    throw std::invalid_argument("Unsupported kernel type");
  }
  cudaFree(l);
  cudaFree(m);

  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  return O;
}

} // namespace flash
