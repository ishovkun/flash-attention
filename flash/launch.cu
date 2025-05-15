#include "KernelParameters.hpp"
#include "common.hpp"
#include "kernel_mma.cuh"
#include "kernel_mma_swizzle.cuh"
#include "kernel_mma_qreg.cuh"
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

__global__ void warp_wmma(const float *Q, const float *K, const float *V,
                               const int N, const int d, const int Bc,
                               const int Br, const float softmax_scale,
                               float *l, float *m, float *O);

__global__ void block_wmma(float const *__restrict__ Q,
                                float const *__restrict__ K,
                                float const *__restrict__ V, int N, int d,
                                int Bc, int Br, float softmax_scale,
                                float *__restrict__ l, float *__restrict__ m,
                                float *__restrict__ O);

__global__ void wmma_rowblock(float const *__restrict__ Q,
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

struct KernelArgs {
 torch::Tensor const *Q, *K, *V;
 float *l, *m;
 torch::Tensor *O;
 float softmax_scale;
 std::unique_ptr<KernelParametersBase> param;
};

template <bool dynamicTileSize>
void launchKernel(KernelArgs & args, auto &&kernel) {
  const int N = args.Q->size(2); // sequence length
  const int d = args.Q->size(3); // head dimension

  int max_sram;
  int requested_sram = args.param->sramSize();
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

  auto const scale = args.softmax_scale;
  auto gridDim = args.param->gridDim();
  auto blockDim = args.param->blockDim();
  if constexpr (!dynamicTileSize)
    kernel<<<gridDim, blockDim, args.param->sramSize()>>>(
        args.Q->data_ptr<float>(), args.K->data_ptr<float>(), args.V->data_ptr<float>(), N, d,
        scale, args.l, args.m, args.O->data_ptr<float>());
  else {
    auto tileSize = args.param->tileSize();
    kernel<<<args.param->gridDim(), args.param->blockDim(), args.param->sramSize()>>>(
        args.Q->data_ptr<float>(), args.K->data_ptr<float>(), args.V->data_ptr<float>(), N, d,
        tileSize.x, tileSize.y, scale, args.l, args.m, args.O->data_ptr<float>());
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

  KernelArgs args {
      .Q = &Q,
      .K = &K,
      .V = &V,
      .l = l,
      .m = m,
      .O = &O,
      .softmax_scale = softmax_scale,
      .param = KernelParametersFactory::create(B, nh, N, d, kernelType),
  };
  auto const tileSize = args.param->tileSize();
  auto const blockDim = args.param->blockDim();

  // std::cout << "Tile size: " << tileSize.x << ", " << tileSize.y <<
  // std::endl; std::cout << "Block dim: " << blockDim.x << ", " << blockDim.y
  // << std::endl; std::cout << "Grid dim: " << gridDim.x << ", " << gridDim.y
  // << ", " << gridDim.z << std::endl;

  // launch kernel
  switch (kernelType) {
  case KernelType::naive1D:
    launchKernel<true>(args, naive_forward_kernel);
    break;
  case KernelType::scalar2D:
    launchKernel<true>(args, forward_kernel_2d);
    break;
  case KernelType::scalar2D_row_tile:
    launchKernel<true>(args, forward_kernel_2d_row_tile);
    break;
  case KernelType::warp_wmma:
    launchKernel<true>(args, warp_wmma);
    break;
  case KernelType::block_wmma:
    launchKernel<true>(args, block_wmma);
    break;
  case KernelType::wmma_row_block:
    launchKernel<true>(args, wmma_rowblock);
    break;
  case KernelType::block_wmma_async:
    launchKernel<true>(args, block_wmma_async);
    break;
  case KernelType::mma: {
    constexpr auto rowsPerBlock = MMAParameters::rowsPerBlock();
    constexpr auto warpsPerBlock = MMAParameters::warpsPerBlock();
    if (constexpr uint32_t colsPerTile = 112; tileSize.x == colsPerTile)
      launchKernel<false>(args, kernel_mma<rowsPerBlock, colsPerTile, warpsPerBlock>);
    else if (constexpr uint32_t colsPerTile = 96; tileSize.x == colsPerTile)
      launchKernel<false>(args, kernel_mma<rowsPerBlock, colsPerTile, warpsPerBlock>);
    else if (constexpr uint32_t colsPerTile = 72; tileSize.x == colsPerTile)
      launchKernel<false>(args, kernel_mma<rowsPerBlock, colsPerTile, warpsPerBlock>);
    else if (constexpr uint32_t colsPerTile = 56; tileSize.x == colsPerTile)
      launchKernel<false>(args, kernel_mma<rowsPerBlock, colsPerTile, warpsPerBlock>);
    else if (constexpr uint32_t colsPerTile = 48; tileSize.x == colsPerTile)
      launchKernel<false>(args, kernel_mma<rowsPerBlock, colsPerTile, warpsPerBlock>);
    else if (constexpr uint32_t colsPerTile = 32; tileSize.x == colsPerTile)
      launchKernel<false>(args, kernel_mma<rowsPerBlock, colsPerTile, warpsPerBlock>);
    else
      unsupportedDimensions(tileSize, blockDim, kernelType);
    break;
  }
  case KernelType::mma_swizzle: {
    constexpr auto rowsPerBlock = MMAParameters::rowsPerBlock();
    constexpr auto warpsPerBlock = MMAParameters::warpsPerBlock();
    std::cout << "Bc = " << tileSize.x << ", Br = " << tileSize.y << std::endl;
    if (constexpr uint32_t colsPerTile = 112; tileSize.x == colsPerTile)
      launchKernel<false>(args, kernel_mma_swizzle<rowsPerBlock, colsPerTile, warpsPerBlock>);
    else if (constexpr uint32_t colsPerTile = 96; tileSize.x == colsPerTile)
      launchKernel<false>(args, kernel_mma_swizzle<rowsPerBlock, colsPerTile, warpsPerBlock>);
    else if (constexpr uint32_t colsPerTile = 72; tileSize.x == colsPerTile)
      launchKernel<false>(args, kernel_mma_swizzle<rowsPerBlock, colsPerTile, warpsPerBlock>);
    else if (constexpr uint32_t colsPerTile = 56; tileSize.x == colsPerTile)
      launchKernel<false>(args, kernel_mma_swizzle<rowsPerBlock, colsPerTile, warpsPerBlock>);
    else if (constexpr uint32_t colsPerTile = 48; tileSize.x == colsPerTile)
      launchKernel<false>(args, kernel_mma_swizzle<rowsPerBlock, colsPerTile, warpsPerBlock>);
    else if (constexpr uint32_t colsPerTile = 32; tileSize.x == colsPerTile)
      launchKernel<false>(args, kernel_mma_swizzle<rowsPerBlock, colsPerTile, warpsPerBlock>);
    else
      unsupportedDimensions(tileSize, blockDim, kernelType);
    break;
  }
  case KernelType::mma_qreg: {
    constexpr auto nw = MMAQregParameters::warpsPerBlock();
    constexpr auto Br = MMAQregParameters::rowsPerTileQ();
    constexpr auto Bc = MMAQregParameters::rowsPerTileK();
    auto tile = args.param->tileSize();
    if (constexpr uint32_t maxHeadDim = 128; tile.z == maxHeadDim)
      launchKernel<false>(args, kernel_mma_qreg<Br, Bc, nw, maxHeadDim>);
    else if (constexpr uint32_t maxHeadDim = 64; tile.z == maxHeadDim)
      launchKernel<false>(args, kernel_mma_qreg<Br, Bc, nw, maxHeadDim>);
    else if (constexpr uint32_t maxHeadDim = 32; tile.z == maxHeadDim)
      launchKernel<false>(args, kernel_mma_qreg<Br, Bc, nw, maxHeadDim>);
    else
      unsupportedDimensions(tileSize, blockDim, kernelType);
    break;
  }
  default: throw std::invalid_argument("Unsupported kernel type");
  }
  cudaFree(l);
  cudaFree(m);

  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  return O;
}

} // namespace flash
