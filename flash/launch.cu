#include "cuda_constants.hpp"
#include "launch.hpp"
#include <thrust/device_vector.h>
#include <iostream>

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

__global__ void warp_wmma_sync(const float *Q, const float *K, const float *V,
                               const int N, const int d, const int Tc,
                               const int Tr, const int Bc, const int Br,
                               const float softmax_scale, float *l, float *m,
                               float *O);

static int maxTileSizeForDeviceSharedMemory(int head_dim) {
  /*
   * We compute the tile size assuming the maximum use of shared memory.
   * This is done by solving a quadratic equation:
   * sh_size = 3*Bc*d + (Bc*Br) -> solve for Bc
   * Assume Bc = Br = x
   * x = (sqrt(9*d*d + 4*m) - 3*d)/2
   */
  int max_sram_size;
  cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
  int const c = max_sram_size / sizeof(float);
  int d = head_dim;
  const int tileSize = floor((sqrt(9. * d * d + 4. * c) - 3. * d) / 2.);
  return tileSize;
}

static int selectTileSize(int d, KernelType kernelType) {
  using namespace flash::constants;
  switch (kernelType) {
  case KernelType::naive1D: {
    return std::min(maxTileSizeForDeviceSharedMemory(d), maxBlockSize);
  }
  case KernelType::scalar2D:
    return std::min(maxTileSizeForDeviceSharedMemory(d),
                    maxBlockSize / warpSize);
  case KernelType::warp_wmma_sync:
    return constants::WMMA_M;
  default:
    throw std::invalid_argument("Unsupported kernel type");
  }
}

static dim3 selectBlockDim(int tileSize, KernelType kernelType) {
  switch (kernelType) {
  case KernelType::naive1D:
    return tileSize;
  case KernelType::scalar2D:
    return dim3(tileSize, constants::warpSize);
  case KernelType::warp_wmma_sync:
    return constants::warpSize;
  default:
    throw std::invalid_argument("Unsupported kernel type");
  }
}

static void checkRequestedSharedMemory(int requested_shared_memory) {
  int max_sram_size;
  cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
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

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                      KernelType kernelType) {
  const int B = Q.size(0);  // batch size
  const int nh = Q.size(1); // number of heads
  const int N = Q.size(2);  // sequence length
  const int d = Q.size(3);  // head dimension

  auto const tileSize = selectTileSize(d, kernelType);
  // std::cout << "Tile size: " << tileSize << std::endl;

  auto const Bc = tileSize;
  auto const Br = tileSize;

  const float softmax_scale = 1.0 / sqrt(d);

  // Initialize O, l, m to HBM
  auto O = torch::zeros_like(Q);
  thrust::device_vector<float> l(B * nh * N, 0.f);
  thrust::device_vector<float> m(B * nh * N, -INFINITY);

  // Calculate SRAM size needed per block
  const int sram_size = (3 * Bc * d * sizeof(float)) + (Bc * Br * sizeof(float));
  checkRequestedSharedMemory(sram_size);

  dim3 gridDim(B, nh); // batch_size x num_heads
  dim3 blockDim = selectBlockDim(tileSize, kernelType);

  // launch kernel
  switch (kernelType) {
  case KernelType::naive1D: {
    const int Tc = ceil((float)N / Bc);
    const int Tr = ceil((float)N / Br);
    naive_forward_kernel<<<gridDim, blockDim, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), N, d, Tc,
        Tr, Bc, Br, softmax_scale, l.data().get(), m.data().get(),
        O.data_ptr<float>());
    break;
  }
  case KernelType::scalar2D:
    forward_kernel_2d<<<gridDim, blockDim, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), N, d, Bc,
        Br, softmax_scale, l.data().get(), m.data().get(),
        O.data_ptr<float>());
    break;
  case KernelType::warp_wmma_sync: {
    const int Tc = ceil((float)N / Bc);
    const int Tr = ceil((float)N / Br);
    warp_wmma_sync<<<gridDim, blockDim, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), N, d, Tc,
        Tr, Bc, Br, softmax_scale, l.data().get(), m.data().get(), O.data_ptr<float>());
    break;
  }
  default:
    throw std::runtime_error("Unsupported kernel type");
  }
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  return O;
}

} // namespace flash
