#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>


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
void forward_kernel2(const float* Q, const float* K, const float* V, const int N, const int d,
                    const int Tc, const int Tr, const int Bc, const int Br, const float softmax_scale,
                    float* l, float *m, float* O)
{
    int tid = threadIdx.x;
    int bx = blockIdx.x; int by = blockIdx.y;  // batch and head index

    // Offset into Q,K,V,O,l,m - different for each batch and head
    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);  // gridDim.y = nh
    int lm_offset = (bx * gridDim.y * N) + (by * N);  // offset for l and m

    // Define SRAM for Q,K,V,S
    extern __shared__ float sram[];
    float* Qi = sram;  // size = Br x d
    float* Kj = &sram[Br*d];     // size = Bc x d
    float* Vj = &sram[d*(Bc+Br)]; // size = Bc x d
    float* S = &sram[d*(Br+2*Bc)];  // size = Br x Bc

    for (int jStart = 0; jStart < N; jStart += Bc) { // loop j tiles

      // load Kj, Vj
      for (int jj = 0; jj < Bc; jj++) {
        auto j = jStart + jj;
        for (int kStart = 0; kStart < d; kStart += blockDim.x) {
          auto k = kStart + tid;
          Kj[jj*d + k] = K[qkv_offset +  j*d + k];
          Vj[jj*d + k] = V[qkv_offset +  j*d + k];
        }
      }

      for (int iStart = 0; iStart < N; iStart += Br) { // loop i tiles

        // Load Qi
        for (int ii = 0; ii < Br; ii++) {
          auto i = iStart + ii;
          for (int kStart = 0; kStart < d; kStart += blockDim.x) {
            auto k = kStart + tid;
            Qi[ii*d + k] = Q[qkv_offset + i*d + k];
          }
        }
        __syncthreads();

        // ------------- parallelize over i now  ---------------
        int ii = tid;
        int i = iStart + ii;

        // Compute product Sij = Qi * Kj 
        // Specifically, Sij = \sum_x (Qix * Kjx)
        float row_m_prev = m[lm_offset + i];
        float row_l_prev = l[lm_offset + i];

        float row_m = -INFINITY;
        for (int jj = 0; jj < Bc; jj++) {
          float Sij = 0.f;
          for (int k = 0; k < d; k++) {
            Sij += Qi[ii*d + k] * Kj[jj*d + k]; 
          }
          Sij *= softmax_scale;
          S[Bc*ii + jj] = Sij;
          row_m = float_max(row_m, Sij);
        }
        __syncthreads();

        float row_l = 0;
        for (int y = 0; y < Bc; y++) {
          S[Bc*ii + y] = __expf(S[Bc*ii + y] - row_m);
          row_l += S[(Bc * tid) + y];
        }
        // Compute new m and l
        float row_m_new = max(row_m_prev, row_m);
        float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_m - row_m_new) * row_l);

        // Write O, l, m to HBM
        for (int x = 0; x < d; x++) {
          float PiyVyx = 0.f;
          for (int y = 0; y < Bc; y++) {
            PiyVyx += S[Bc*ii + y] * Vj[y*d + x];
          }
          O[qkv_offset + i*d + x] = ((row_l_prev * __expf(row_m_prev - row_m_new) * O[qkv_offset + i*d + x])  +
                                     (__expf(row_m - row_m_new) * PiyVyx)) / row_l_new;
        }
        m[lm_offset + i] = row_m_new;
        l[lm_offset + i] = row_l_new;
      }
      __syncthreads();
    }
}

__global__
void forward_kernel_2d(const float* Q, const float* K, const float* V, const int N, const int d,
                       const int Tc, const int Tr, const int Bc, const int Br, const float softmax_scale,
                       float* l, float *m, float* O)
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
    for (int kStart = 0; kStart < d; kStart += blockDim.x) {
      auto k = kStart + tx;
      auto jj = ty;
      auto j = jStart + jj;
      Kj[jj*d + k] = (j < N) ? K[qkv_offset +  j*d + k] : 0.f;
      Vj[jj*d + k] = (j < N) ? V[qkv_offset +  j*d + k] : 0.f;
    }
    
    for (int iStart = 0; iStart < N; iStart += Br) { // loop i tiles
      // Load Qi
      auto ii = ty;
      auto i = iStart + ii;  
      for (int kStart = 0; kStart < d; kStart += blockDim.x) {
        auto k = kStart + tx;
        Qi[ii*d + k] = (i < N) ? Q[qkv_offset + i*d + k] : 0.f;
      }
      // __syncthreads(); -- not needed 
                                             
      float row_m = -INFINITY;
      for (int jj = tx; jj < Bc; jj += blockDim.x) {
        float Sij = 0.f;
        for (int k = 0; k < d; k++) {
          Sij += Qi[ii*d + k] * Kj[jj*d + k]; 
        }
        Sij *= softmax_scale;
        S[Bc*ii + jj] = Sij;
        row_m = float_max(row_m, Sij);
      }
      row_m = warpReduce<float_max>(row_m);
      
      // S = [Br x Bc]
      float row_l = 0.f;
      for (int jj = tx; jj < Bc; jj += blockDim.x) {
        float Sij = __expf(S[Bc*ii + jj] - row_m);
        S[Bc*ii + jj] = Sij;
        row_l += Sij;
      }
      row_l = warpReduce<float_add>(row_l); 

      float row_m_prev = m[lm_offset + i];
      float row_l_prev = l[lm_offset + i];
      float row_m_new = float_max(row_m_prev, row_m);
      float row_l_new = __expf(row_m_prev - row_m_new) * row_l_prev +
                        __expf(row_m - row_m_new) * row_l;
      
      // __syncthreads(); -- no need <= there is only one warp in j direction

      // Product Oik = Pin * Vnk
      // O[Br,d] = S[Br, Bc] * V[Bc, d]
      for (int k = tx; k < d; k += blockDim.x) {
        float PinVnk = 0.f;
        for (int n = 0; n < Bc; n++) {
          PinVnk += S[Bc*ii + n] * Vj[n*d + k];
        }
        if (i < N)
          O[qkv_offset + i*d + k] = ((row_l_prev * __expf(row_m_prev - row_m_new) * O[qkv_offset + i*d + k])  +
                                     (__expf(row_m - row_m_new) * PinVnk)) / row_l_new;
      }
      // save new l and m
      if (tx == 0) {
        m[lm_offset + i] = row_m_new;
        l[lm_offset + i] = row_l_new;
      }
    } // end i-tile loop
    
    __syncthreads();
  }
}

torch::Tensor forward_opt(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    // TODO: determine Bc, Br dynamically
    const int Bc = 32; const int Br = 32;

    const int B = Q.size(0); const int nh = Q.size(1);
    const int N = Q.size(2); const int d = Q.size(3);

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
    constexpr int warpSize = 32;
    dim3 block_dim(warpSize, max(Br, Bc));  // Bc threads per block
    // dim3 block_dim(max(Br, Bc));  // Bc threads per block

    forward_kernel_2d<<<grid_dim, block_dim, sram_size>>>(
    // forward_kernel2<<<grid_dim, block_dim, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        N, d, Tc, Tr, Bc, Br, softmax_scale,
        l.data_ptr<float>(), m.data_ptr<float>(), O.data_ptr<float>()
    );
    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));
    }
    return O;
}
