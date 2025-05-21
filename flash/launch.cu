#include "KernelParameters.hpp"
#include "common.hpp"
#include "kernel_mma.cuh"
#include "kernel_mma_swizzle.cuh"
#include "kernel_mma_qreg.cuh"
#include "kernel_mma_qreg_f32x4load.cuh"
#include "launch.hpp"
#include "wmma.hpp"
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace flash {

__global__ void naive_forward_kernel(float const *Q, float const *K,
                                     float const *V, int N, int d,
                                     const int Bc, int Br,
                                     float softmax_scale, float *l,
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

void unsupportedDimensions(dim3 tileDim, dim3 blockDim, KernelType kernel) {
  std::ostringstream err;
  err << "error : " << "Unsupported tile size: " << tileDim.x << ", "
      << tileDim.y << " and block size: " << blockDim.x << ", " << blockDim.y
      << " for kernel type: " << to_string(kernel);
  throw std::invalid_argument(err.str());
}

/*
These crazy template thingies are needed in launch_kernel function
to launch various kernels depending on the arumments that they take.
*/

// Naive kernel signature
using DynamicTileSizeKernelSignature = void (*) (
  float const*, float const*, float const*,
  int, int, int, int, float,
  float*, float*, float*);

// Faster kernels use static tile sized
using StaticTileSizeKernelWithLMSignature = void (*) (
  float const*, float const*, float const*,
  int, int, float,
  float*, float*, float*);

// Most optimized kenrle don't really need l and m parameters
using StaticTileSizeKernelSignature = void (*) (
  float const*, float const*, float const*,
  int, int, float, float*);

template <typename F>
concept DynamicTileSizeKernel = std::is_convertible_v<F, DynamicTileSizeKernelSignature>;

template <typename F>
concept StaticTileSizeKernelGlobalLM = std::is_convertible_v<F, StaticTileSizeKernelWithLMSignature>;

template <typename F>
concept StaticTileSizeKernelLocalLM = std::is_convertible_v<F, StaticTileSizeKernelSignature>;

struct KernelArgs {
 torch::Tensor const *Q, *K, *V;
 torch::Tensor *O;
 float softmax_scale;
 std::unique_ptr<KernelParametersBase> param;
};

template <typename F, typename T, T... Vs>
constexpr void static_for(std::integer_sequence<T, Vs...>, F&& f) {
  (f(std::integral_constant<T, Vs>{}), ...);
}

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

  float *l = nullptr, *m = nullptr;
  if constexpr (!StaticTileSizeKernelLocalLM<decltype(kernel)>) {
    auto const & Q = *args.Q;
    const int B = Q.size(0);  // batch size
    const int nh = Q.size(1); // number of heads
    cudaMalloc(&l, B * nh * N * sizeof(float));
    cudaMalloc(&m, B * nh * N * sizeof(float));
  }

  auto const scale = args.softmax_scale;
  auto gridDim = args.param->gridDim();
  auto blockDim = args.param->blockDim();
  if constexpr (DynamicTileSizeKernel<decltype(&kernel)>) {
    auto tileSize = args.param->tileSize();
    kernel<<<args.param->gridDim(), args.param->blockDim(), args.param->sramSize()>>>(
          args.Q->data_ptr<float>(), args.K->data_ptr<float>(), args.V->data_ptr<float>(), N, d,
          tileSize.x, tileSize.y, scale, l, m, args.O->data_ptr<float>());
  }
  else if constexpr (StaticTileSizeKernelGlobalLM<decltype(kernel)>) {
    kernel<<<gridDim, blockDim, args.param->sramSize()>>>(
        args.Q->data_ptr<float>(), args.K->data_ptr<float>(), args.V->data_ptr<float>(), N, d,
        scale, l, m, args.O->data_ptr<float>());
  }
  else if constexpr (StaticTileSizeKernelLocalLM<decltype(kernel)>) {
    kernel<<<gridDim, blockDim, args.param->sramSize()>>>(
        args.Q->data_ptr<float>(), args.K->data_ptr<float>(), args.V->data_ptr<float>(), N, d,
        scale, args.O->data_ptr<float>());
  }
  else {
    static_assert([] { return false; }(), "Unsupported kernel signature :-(");
  }
  if (l) cudaFree(l);
  if (m) cudaFree(m);
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

  KernelArgs args {
      .Q = &Q,
      .K = &K,
      .V = &V,
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
    launchKernel(args, naive_forward_kernel);
    break;
  case KernelType::scalar2D:
    launchKernel(args, forward_kernel_2d);
    break;
  case KernelType::scalar2D_row_tile:
    launchKernel(args, forward_kernel_2d_row_tile);
    break;
  case KernelType::warp_wmma:
    launchKernel(args, warp_wmma);
    break;
  case KernelType::block_wmma:
    launchKernel(args, block_wmma);
    break;
  case KernelType::wmma_row_block:
    launchKernel(args, wmma_rowblock);
    break;
  case KernelType::block_wmma_async:
    launchKernel(args, block_wmma_async);
    break;
  case KernelType::mma: {
    constexpr auto validRowsPerTileK = std::integer_sequence<uint32_t, 16, 32, 48, 56, 72, 96, 112>{};
    std::cout << "launching : " << "tile size: Bc =" << tileSize.x << ", Br = "
              << tileSize.y << " and block size: " << blockDim.x << ", " << blockDim.y
              << " for kernel type: " << to_string(kernelType) << std::endl;

    bool launched = false;
    static_for(validRowsPerTileK, [&](auto rowsPerTileK) {
      if (tileSize.x == rowsPerTileK) {
        constexpr auto rowsPerTileQ = MMAParameters::rowsPerBlock();
        constexpr auto warpsPerBlock = MMAParameters::warpsPerBlock();
        launchKernel(args, kernel_mma<rowsPerTileQ, rowsPerTileK, warpsPerBlock>);
        launched = true;
      }
    });
    if (!launched)  { unsupportedDimensions(tileSize, blockDim, kernelType); }
    break;
  }
  case KernelType::mma_swizzle: {
    constexpr auto validRowsPerTileK = std::integer_sequence<uint32_t, 16, 32, 48, 56, 72, 96, 112>{};
    bool launched = false;
    static_for(validRowsPerTileK, [&](auto rowsPerTileK) {
      if (tileSize.x == rowsPerTileK) {
        constexpr auto rowsPerTileQ = MMAParameters::rowsPerBlock();
        constexpr auto warpsPerBlock = MMAParameters::warpsPerBlock();
        launchKernel(args, kernel_mma_swizzle<rowsPerTileQ, rowsPerTileK, warpsPerBlock>);
        launched = true;
      }
    });
    if (!launched)  { unsupportedDimensions(tileSize, blockDim, kernelType); }
    break;
  }
  case KernelType::mma_qreg: {
    if (d % 2 != 0) {
      throw std::invalid_argument("Head dim must be even for kernel" + to_string(kernelType));
    }
    auto tile = args.param->tileSize();
    constexpr auto validMaxHeadDim = std::integer_sequence<uint32_t, 32, 64, 96, 128>{};
    bool launched = false;
    static_for(validMaxHeadDim, [&](auto maxHeadDim) {
      if (tileSize.z == maxHeadDim) {
        constexpr auto Br = MMAQregParameters::rowsPerTileQ();
        constexpr auto Bc = MMAQregParameters::rowsPerTileK();
        launchKernel(args, kernel_mma_qreg<Br, Bc, maxHeadDim>);
        launched = true;
      }
    });
    if (!launched)  { unsupportedDimensions(tileSize, blockDim, kernelType); }
    break;
  }
  case KernelType::mma_qreg_f32x4load: {
    if (d % 2 != 0) {
      throw std::invalid_argument("Head dim must be even for kernel" + to_string(kernelType));
    }
    auto tile = args.param->tileSize();
    constexpr auto validMaxHeadDim = std::integer_sequence<uint32_t, 32, 64, 96, 128>{};
    bool launched = false;
    static_for(validMaxHeadDim, [&](auto maxHeadDim) {
      if (tileSize.z == maxHeadDim) {
        constexpr auto Br = MMAQregParameters::rowsPerTileQ();
        constexpr auto Bc = MMAQregParameters::rowsPerTileK();
        launchKernel(args, kernel_mma_qreg_f32x4load<Br, Bc, maxHeadDim>);
        launched = true;
      }
    });
    if (!launched)  { unsupportedDimensions(tileSize, blockDim, kernelType); }
    break;
  }
  default: throw std::invalid_argument("Unsupported kernel type");
  }

  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  return O;
}

} // namespace flash
