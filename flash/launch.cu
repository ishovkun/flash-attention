#include "launch.hpp"
#include <iostream>

namespace flash {

__global__ void naive_forward_kernel(const float *Q, const float *K,
                                     const float *V, const int N, const int d,
                                     const int Tc, const int Tr, const int Bc,
                                     const int Br, const float softmax_scale,
                                     float *l, float *m, float *O);

static int getTileSize(int head_dim) {
  int max_sram_size;
  cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
  int const c = max_sram_size / sizeof(float);
  int d = head_dim;
  const int tileSize = floor((sqrt(9. * d * d + 4. * c) - 3. * d) / 2.);
  return tileSize;
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

__global__ void forward_kernel_2d(float const *__restrict__ Q,
                                  float const *__restrict__ K,
                                  float const *__restrict__ V, int N, int d,
                                  int Tc, int Tr, int Bc, int Br,
                                  float softmax_scale, float *__restrict__ l,
                                  float *__restrict__ m, float *__restrict__ O);

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
                      KernelType kernel_type) {
  // TODO: determine Bc, Br dynamically
  // const int Bc = 32; const int Br = 32;

  const int B = Q.size(0);  // batch size
  const int nh = Q.size(1); // number of heads
  const int N = Q.size(2);  // sequence length
  const int d = Q.size(3);  // head dimension

  /*
   * sh_size //
   * sh_size = 3*Bc*d + (Bc*Br)
   * Assume Bc = Br = T (tile_size)
   * x = (sqrt(9*d*d + 4*m) - 3*d)/2
   */
  // Compute tile size using maximum shared memory per block
  auto tileSize = getTileSize(d);
  // tileSize = 52;
  // std::cout << "Tile size: " << tileSize << std::endl;

  // Limit tile size since only 1024 threads can be launched per block
  constexpr int maxBlockSize = 1024;
  if (kernel_type == KernelType::naive1D) {
    tileSize = min(tileSize, maxBlockSize);
  } else {
    constexpr int warpSize = 32;
    auto maxTileSize = maxBlockSize / warpSize;
    tileSize = std::min(tileSize, maxTileSize);
  }
  // std::cout << "Tile size: " << tileSize << std::endl;

  auto const Bc = tileSize;
  auto const Br = tileSize;

  const int Tc = ceil((float)N / Bc);
  const int Tr = ceil((float)N / Br);
  const float softmax_scale = 1.0 / sqrt(d);

  // Initialize O, l, m to HBM
  auto O = torch::zeros_like(Q);
  auto l = torch::zeros({B, nh, N});
  auto m = torch::full({B, nh, N}, -INFINITY);
  torch::Device device(torch::kCUDA);
  l = l.to(device);
  m = m.to(device);

  // Calculate SRAM size needed per block
  const int sram_size =
      (3 * Bc * d * sizeof(float)) + (Bc * Br * sizeof(float));
  checkRequestedSharedMemory(sram_size);

  dim3 grid_dim(B, nh); // batch_size x num_heads
  dim3 block_dim(Bc);   // Bc threads per block
  if (kernel_type == KernelType::scalar2D) {
    constexpr int warpSize = 32;
    block_dim = dim3(warpSize, std::max(Br, Bc));
  }
  // std::cout << "Block size: " << block_dim.x << "x" << block_dim.y << std::endl;

  // launch kernel
  switch (kernel_type) {
  case KernelType::naive1D:
    naive_forward_kernel<<<grid_dim, block_dim, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), N, d, Tc,
        Tr, Bc, Br, softmax_scale, l.data_ptr<float>(), m.data_ptr<float>(),
        O.data_ptr<float>());
    break;
  case KernelType::scalar2D:
    forward_kernel_2d<<<grid_dim, block_dim, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), N, d, Tc,
        Tr, Bc, Br, softmax_scale, l.data_ptr<float>(), m.data_ptr<float>(),
        O.data_ptr<float>());
    break;
  default:
    throw std::runtime_error("Unsupported kernel type");
  }
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  return O;
}

} // namespace flash
