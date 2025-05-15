#include <climits>
#include <cmath>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "common.hpp"

namespace flash {

__global__
void naive_forward_kernel(const float* Q, const float* K, const float* V, const int N, const int d,
                          const int Bc, const int Br, const float softmax_scale,
                          float* l, float *m, float* O) {
    int tx = threadIdx.x;
    int batch = blockIdx.x;
    int head = blockIdx.y;
    auto numHeads = gridDim.y;

    auto const Tc = common::ceil_div(N, Bc);
    auto const Tr = common::ceil_div(N, Br);

    // Offset into Q,K,V,O,l,m - different for each batch and head
    int qkv_offset = (batch * numHeads * N * d) + (head * N * d);
    int lm_offset = (batch * numHeads * N) + (head * N);

    // Define SRAM for Q,K,V,S
    extern __shared__ float sram[];
    int tile_size = Bc * d;  // size of Qi, Kj, Vj
    float* Qi = sram;
    float* Kj = &sram[tile_size];     // [tile_size]
    float* Vj = &sram[tile_size * 2]; // [tile_size]
    float* S = &sram[tile_size * 3];  // [tile_size]

    // set l and m to default values
    for (int i = tx; i < N; i += blockDim.x) {
      l[lm_offset + i] = 0.f;
      m[lm_offset + i] = -INFINITY;
    }

    for (int j = 0; j < Tc; j++) { // j tile
      //
      // Potentially cropped Bc in the last tile
      auto Bcc = min(Bc, N - j*Bc);
      // Load Kj, Vj to SRAM
      for (int x = 0; x < d; x++) {
          // K_jx
          // true jj = (Bc*d*j) + (tx * d)
          Kj[(tx * d) + x] = (j*Bc + tx < N) ? K[qkv_offset + (tile_size * j) + (tx * d) + x] : 0.f;
          Vj[(tx * d) + x] = (j*Bc + tx < N) ? V[qkv_offset + (tile_size * j) + (tx * d) + x] : 0.f;
      }
      __syncthreads();

        for (int i = 0; i < Tr; i++)  { // i tile
            // true ii = (Br*d*i) + (tx * d)

            // Load Qi to SRAM, l and m to registers
            for (int x = 0; x < d; x++) {
                Qi[(tx * d) + x] = (i*Br + tx < N) ? Q[qkv_offset + (tile_size * i) + (tx * d) + x] : 0.f;
            }
            float row_m_prev = (Br*i + tx < N) ? m[lm_offset + (Br * i) + tx] : -INFINITY;
            float row_l_prev = (Br*i + tx < N) ? l[lm_offset + (Br * i) + tx] : 0.f;

            // S = QK^T, row_m = rowmax(S)
            float row_m = -INFINITY;
            for (int y = 0; y < Bcc; y++) {
                float sum = 0;
                for (int x = 0; x < d; x++) {
                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                }
                sum *= softmax_scale;
                S[(Bc * tx) + y] = sum;

                if (sum > row_m)
                    row_m = sum;
            }

            // P = exp(S - row_m), row_l = rowsum(P)
            float row_l = 0;
            for (int y = 0; y < Bcc; y++) {
                S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - row_m);
                row_l += S[(Bc * tx) + y];
            }

            // Compute new m and l
            float row_m_new = max(row_m_prev, row_m);
            float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_m - row_m_new) * row_l);

            // Write O, l, m to HBM
            for (int x = 0; x < d; x++) {
                float pv = 0;  // Pij * Vj
                for (int y = 0; y < Bcc; y++) {
                    pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
                }
                if (i*Br + tx < N)
                  O[qkv_offset + (tile_size * i) + (tx * d) + x] = (1 / row_l_new) \
                      * ((row_l_prev * __expf(row_m_prev - row_m_new) * O[qkv_offset + (tile_size * i) + (tx * d) + x]) \
                      + (__expf(row_m - row_m_new) * pv));
            }
            if (i*Br + tx < N) {
              m[lm_offset + (Bc * i) + tx] = row_m_new;
              l[lm_offset + (Bc * i) + tx] = row_l_new;
            }
        }
        __syncthreads();
    }
}


}
