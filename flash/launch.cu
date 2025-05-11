#include "KernelParameters.hpp"
#include "common.hpp"
#include "kernel_mma_sync.cuh"
#include "launch.hpp"
#include "wmma.hpp"
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace flash {

__global__ void naive_forward_kernel(const float *Q, const float *K,
                                     const float *V, const int N, const int d,
                                     const int Bc, const int Br,
                                     const float softmax_scale, float *l,
                                     float *m, float *O);

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
                               const int N, const int d, const int Bc,
                               const int Br, const float softmax_scale,
                               float *l, float *m, float *O);

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

__global__ void block_wmma_async(float const *__restrict__ Q,
                                 float const *__restrict__ K,
                                 float const *__restrict__ V, int N, int d,
                                 int Bc, int Br, float softmax_scale,
                                 float *__restrict__ l, float *__restrict__ m,
                                 float *__restrict__ O);

// static void checkRequestedSharedMemory(int requested_shared_memory) {
//   int max_sram_size;
//   cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock,
//   0); if (requested_shared_memory > max_sram_size) {
//     std::cerr << "Warning: Requested shared memory " <<
//     requested_shared_memory
//               << " exceeds the default maximum " << max_sram_size <<
//               std::endl;
//     // std::cout << "Trying to bump up the maximum: " << std::endl;
//     // std::cerr << "Requested shared memory " << requested_shared_memory
//     //           << " exceeds maximum allowed (" << max_sram_size << ")"
//     //           << std::endl;
//     throw std::runtime_error("Requested shared memory exceeds maximum
//     allowed");
//   }
// }

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

// #define UNSUPPORTED_DIMENSIONS(msg)                                            \
//   {                                                                            \
//     std::ostringstream err;                                                    \
//     err << __FILE__ << "(" << __LINE__ << ") "                                 \
//         << "error : " << msg;                                                  \
//     throw std::invalid_argument(err.str());                                    \
//   }
void unsupportedDimensions(dim3 tileDim, dim3 blockDim, KernelType kernel) {
  std::ostringstream err;
  err << "error : " << "Unsupported tile size: " << tileDim.x << ", "
      << tileDim.y << " and block size: " << blockDim.x << ", " << blockDim.y
      << " for kernel type: " << static_cast<int>(kernel);
  throw std::invalid_argument(err.str());
}

template <bool dynamicTileSize>
void launchKernel(torch::Tensor const &Q, torch::Tensor const &K,
                  torch::Tensor const &V, auto *l, auto *m,
                  torch::Tensor const &O, auto scale, auto &param,
                  auto &&kernel) {
  const int N = Q.size(2); // sequence length
  const int d = Q.size(3); // head dimension

  int max_sram;
  int requested_sram = param.sramSize();
  cudaDeviceGetAttribute(&max_sram, cudaDevAttrMaxSharedMemoryPerBlock, 0);
  if (requested_sram > max_sram) {
    std::cerr << "Warning: Requested shared memory " << requested_sram
              << " exceeds the default maximum " << max_sram << std::endl;
    std::cerr << "Trying to bump up the maximum: " << std::endl;
    auto check = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, requested_sram);
    if (check != cudaSuccess) {
      std::cerr << "Failed to set dynamic shared memory size: "
                << requested_sram << std::endl;
      throw std::runtime_error("Failed to set dynamic shared memory size");
    }
  }

  if constexpr (!dynamicTileSize)
    kernel<<<param.gridDim(), param.blockDim(), param.sramSize()>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), N, d,
        scale, l, m, O.data_ptr<float>());
  else {
    auto tileSize = param.tileSize();
    kernel<<<param.gridDim(), param.blockDim(), param.sramSize()>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), N, d,
        tileSize.x, tileSize.y, scale, l, m, O.data_ptr<float>());
  }
}

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                      KernelType kernelType) {
  const int B = Q.size(0);  // batch size
  const int nh = Q.size(1); // number of heads
  const int N = Q.size(2);  // sequence length
  const int d = Q.size(3);  // head dimension

  const float softmax_scale = 1.0 / sqrt(d);

  // Initialize O, l, m to HBM
  auto O = torch::zeros_like(Q);

  float *l, *m;
  cudaMalloc(&l, B * nh * N * sizeof(float));
  cudaMalloc(&m, B * nh * N * sizeof(float));

  // This parameter is only used in templated kernels
  auto kernelParams = FlashKernelParametersFactory::create(B, nh, N, d, kernelType);

  auto const tileSize = kernelParams->tileSize();
  // auto const Br = tileSize.y;
  // auto const Bc = tileSize.x;

  // auto const gridDim = kernelParams->gridDim();
  auto const blockDim = kernelParams->blockDim();

  // std::cout << "Tile size: " << tileSize.x << ", " << tileSize.y <<
  // std::endl; std::cout << "Block dim: " << blockDim.x << ", " << blockDim.y
  // << std::endl; std::cout << "Grid dim: " << gridDim.x << ", " << gridDim.y
  // << ", " << gridDim.z << std::endl;

  // launch kernel
  switch (kernelType) {
  case KernelType::naive1D:
    launchKernel<true>(Q, K, V, l, m, O, softmax_scale, *kernelParams,
                       naive_forward_kernel);
    break;
  case KernelType::scalar2D:
    launchKernel<true>(Q, K, V, l, m, O, softmax_scale, *kernelParams,
                       forward_kernel_2d);
    break;
  case KernelType::scalar2D_row_tile:
    launchKernel<true>(Q, K, V, l, m, O, softmax_scale, *kernelParams,
                       forward_kernel_2d_row_tile);
    break;
  case KernelType::warp_wmma_sync:
    launchKernel<true>(Q, K, V, l, m, O, softmax_scale, *kernelParams,
                       warp_wmma_sync);
    break;
  case KernelType::block_wmma_sync:
    launchKernel<true>(Q, K, V, l, m, O, softmax_scale, *kernelParams,
                       block_wmma_sync);
    break;
  case KernelType::wmma_sync_row_block:
    launchKernel<true>(Q, K, V, l, m, O, softmax_scale, *kernelParams,
                       wmma_sync_rowblock);
    break;
  case KernelType::block_wmma_async:
    launchKernel<true>(Q, K, V, l, m, O, softmax_scale, *kernelParams,
                       block_wmma_async);
    break;
  case KernelType::mma_sync: {
    std::cout << "tile size: Br = " << tileSize.y << " Bc = " << tileSize.x << std::endl;
    constexpr auto rowsPerBlock = MMASyncKernelParameters::rowsPerBlock();
    constexpr auto warpsPerBlock = MMASyncKernelParameters::warpsPerBlock();
    if (constexpr uint32_t colsPerBlock = 32; tileSize.x == colsPerBlock) {
      launchKernel<false>(Q, K, V, l, m, O, softmax_scale, *kernelParams,
                          kernel_mma_sync<rowsPerBlock, colsPerBlock, warpsPerBlock>);
    }
    else if (constexpr uint32_t colsPerBlock = 96; tileSize.x == colsPerBlock) {
      launchKernel<false>(Q, K, V, l, m, O, softmax_scale, *kernelParams,
                          kernel_mma_sync<rowsPerBlock, colsPerBlock, warpsPerBlock>);
    }
    else if (constexpr uint32_t colsPerBlock = 56; tileSize.x == colsPerBlock) {
      launchKernel<false>(Q, K, V, l, m, O, softmax_scale, *kernelParams,
                          kernel_mma_sync<rowsPerBlock, colsPerBlock, warpsPerBlock>);
    }
    else if (constexpr uint32_t colsPerBlock = 48; tileSize.x == colsPerBlock) {
      launchKernel<false>(Q, K, V, l, m, O, softmax_scale, *kernelParams,
                          kernel_mma_sync<rowsPerBlock, colsPerBlock, warpsPerBlock>);
    }
    else {
      unsupportedDimensions(tileSize, blockDim, kernelType);
    }
  } break;
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
