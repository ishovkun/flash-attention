#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace flash {

inline float __device__ float_max(float a, float b) {
  return a > b ? a : b;
}

inline float __device__ float_add(float a, float b) {
  return a + b;
}

template <auto binaryFunc = float_add>
__device__ float warpReduce(float value) {
  constexpr int warpSize = 32;
  int lane = threadIdx.x % warpSize;
  for (int s = warpSize / 2; s > 0; s >>= 1) {
    auto tmp = __shfl_down_sync(UINT32_MAX, value, s);
    if (lane < s) value = binaryFunc(value, tmp);
  }
  return __shfl_sync(UINT32_MAX, value, 0);
}

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
                       float* __restrict__ O)
{
  int bx = blockIdx.x;  // batch index
  int by = blockIdx.y;  // head index

  int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);  // gridDim.y = nh
  int lm_offset = (bx * gridDim.y * N) + (by * N);  // offset for l and m

  extern __shared__ float sram[];
  float* Qi = sram;  // size = Br x d
  float* Kj = &sram[Br*d];     // size = Bc x d
  float* Vj = &sram[d*(Bc+Br)]; // size = Bc x d
  float* S = &sram[d*(Br+2*Bc)];  // size = Br x Bc

  auto tx = threadIdx.x;
  auto ty = threadIdx.y;

  for (int jStart = 0; jStart < N; jStart += Bc) { // loop j tiles
    // load Kj, Vj
    for (int k = tx; k < d; k += blockDim.x) {
      auto jj = ty;
      auto j = jStart + jj;
      auto inBounds = j < N;
      Kj[jj*d + k] = inBounds ? K[qkv_offset +  j*d + k] : 0.f;
      Vj[jj*d + k] = inBounds ? V[qkv_offset +  j*d + k] : 0.f;
    }
    // __syncthreads();

    for (int iStart = 0; iStart < N; iStart += Br) { // loop i tiles
      // Bc might be smaller in the last tile cause N does not necessarily divide by Bc
      // then the local size of Bc must change to prevent index errors
      int const Bc_cur = min(Bc, N - iStart);
      // Load Qi
      auto ii = ty;
      auto i = iStart + ii;
      for (int k = tx; k < d; k += blockDim.x)
          Qi[ii*d + k] = (i < N) ? Q[qkv_offset + i*d + k] : 0.f;
      // __syncthreads();// -- not needed

      // Compute Sij and row_max
      float row_m = -INFINITY;
      for (int jj = tx; jj < Bc_cur; jj += blockDim.x) {
        float Sij = 0.f;
        for (int k = 0; k < d; k++) {
          Sij += Qi[ii*d + k] * Kj[jj*d + k];
        }
        Sij *= softmax_scale;
        S[Bc_cur*ii + jj] = Sij;
        row_m = float_max(row_m, Sij);
      }
      row_m = warpReduce<float_max>(row_m);

      // S = [Br x Bc]
      float row_l = 0.f;
      for (int jj = tx; jj < Bc_cur; jj += blockDim.x) {
        float Sij = __expf(S[Bc_cur*ii + jj] - row_m);
        S[Bc_cur*ii + jj] = Sij;
        row_l += Sij;
      }
      row_l = warpReduce<float_add>(row_l);

      float row_m_prev = (i < N) ? m[lm_offset + i] : -INFINITY;
      float row_l_prev = (i < N) ? l[lm_offset + i] : 0.f;
      float row_m_new = float_max(row_m_prev, row_m);
      float row_l_new = __expf(row_m_prev - row_m_new) * row_l_prev + __expf(row_m - row_m_new) * row_l;

      // __syncthreads(); -- no need <= there is only one warp in j direction

      // Product Oik = Pin * Vnk
      // O[Br,d] = S[Br, Bc] * V[Bc, d]
      for (int k = tx; k < d; k += blockDim.x) {
        float PinVnk = 0.f;
        for (int n = 0; n < Bc_cur; n++) {
          PinVnk += S[Bc_cur*ii + n] * Vj[n*d + k];
        }
        if (i < N)
          O[qkv_offset + i*d + k] = ((row_l_prev * __expf(row_m_prev - row_m_new) * O[qkv_offset + i*d + k])  +
                                     (__expf(row_m - row_m_new) * PinVnk)) / row_l_new;
      }
      // save new l and m
      if (tx == 0 && i < N) {
        m[lm_offset + i] = row_m_new;
        l[lm_offset + i] = row_l_new;
      }
    } // end i-tile loop

    __syncthreads();
  }
}

// torch::Tensor forward_opt(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
//     // TODO: determine Bc, Br dynamically
//     const int Bc = 32; const int Br = 32;

//     const int B = Q.size(0); const int nh = Q.size(1);
//     const int N = Q.size(2); const int d = Q.size(3);

//     const int Tc = ceil((float) N / Bc); const int Tr = ceil((float) N / Br);
//     const float softmax_scale = 1.0 / sqrt(d);

//     // Initialize O, l, m to HBM
//     auto O = torch::zeros_like(Q);
//     auto l = torch::zeros({B, nh, N});
//     auto m = torch::full({B, nh, N}, -INFINITY);
//     torch::Device device(torch::kCUDA);
//     l = l.to(device); m = m.to(device);

//     // Calculate SRAM size needed per block
//     const int sram_size = (3 * Bc * d * sizeof(float)) + (Bc * Br * sizeof(float));
//     int max_sram_size;
//     cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
//     printf("Max shared memory: %d, requested shared memory: %d \\n", max_sram_size, sram_size);

//     dim3 grid_dim(B, nh);  // batch_size x num_heads
//     constexpr int warpSize = 32;
//     dim3 block_dim(warpSize, max(Br, Bc));  // Bc threads per block
//     // dim3 block_dim(max(Br, Bc));  // Bc threads per block

//     forward_kernel_2d<<<grid_dim, block_dim, sram_size>>>(
//     // forward_kernel2<<<grid_dim, block_dim, sram_size>>>(
//         Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
//         N, d, Tc, Tr, Bc, Br, softmax_scale,
//         l.data_ptr<float>(), m.data_ptr<float>(), O.data_ptr<float>()
//     );
//     cudaDeviceSynchronize();
//     if (cudaGetLastError() != cudaSuccess) {
//         printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));
//     }
//     return O;
// }

}
