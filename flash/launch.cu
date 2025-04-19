#include "launch.hpp"



namespace flash {

__global__
void naive_forward_kernel(const float* Q, const float* K, const float* V, const int N, const int d,
                          const int Tc, const int Tr, const int Bc, const int Br, const float softmax_scale,
                          float* l, float *m, float* O);

__global__
void forward_kernel_2d(float const * __restrict__ Q,
                       float const * __restrict__ K,
                       float const * __restrict__ V,
                       int N, int d,
                       int Tc, int Tr,
                       int Bc, int Br,
                       float softmax_scale,
                       float* __restrict__ l,
                       float* __restrict__ m,
                       float* __restrict__ O);

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                      KernelType kernel_type) {
  // TODO: determine Bc, Br dynamically
  const int Bc = 32; const int Br = 32;

  const int B = Q.size(0); // batch size
  const int nh = Q.size(1); // number of heads
  const int N = Q.size(2); // sequence length
  const int d = Q.size(3); // head dimension

  const int Tc = ceil((float) N / Bc); const int Tr = ceil((float) N / Br);
  const float softmax_scale = 1.0 / sqrt(d);

  // Initialize O, l, m to HBM
  auto O = torch::zeros_like(Q);
  auto l = torch::zeros({B, nh, N});
  auto m = torch::full({B, nh, N}, -INFINITY);
  torch::Device device(torch::kCUDA);
  l = l.to(device); m = m.to(device);

  // Calculate SRAM size needed per block
  const int sram_size = (3 * Bc * d * sizeof(float)) + (Bc * Br * sizeof(float));
  int max_sram_size;
  cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
  printf("Max shared memory: %d, requested shared memory: %d \\n", max_sram_size, sram_size);

  dim3 grid_dim(B, nh);  // batch_size x num_heads
  dim3 block_dim(Bc);  // Bc threads per block
  if (kernel_type == KernelType::scalar2D) {
    constexpr int warpSize = 32;
    block_dim = dim3(warpSize, max(Br, Bc));
  }
  // launch kernel
  switch (kernel_type) {
    case KernelType::naive1D:
      naive_forward_kernel<<<grid_dim, block_dim, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        N, d, Tc, Tr, Bc, Br, softmax_scale,
        l.data_ptr<float>(), m.data_ptr<float>(), O.data_ptr<float>()
      );
      break;
    case KernelType::scalar2D:
      forward_kernel_2d<<<grid_dim, block_dim, sram_size>>>(
          Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
          N, d, Tc, Tr, Bc, Br, softmax_scale,
          l.data_ptr<float>(), m.data_ptr<float>(), O.data_ptr<float>()
      );
      break;
    default:
      throw std::runtime_error("Unsupported kernel type");
  }

  return O;
}

}
