#include "common.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream> // debug

namespace flash {

__global__ void forward_kernel_2d_row_tile(
    float const *__restrict__ Q, // query vector
    float const *__restrict__ K, // key vector
    float const *__restrict__ V, // value vector
    int N,                       // sequence length
    int d,                       // head_dim
    int Bc, int Br,              // column tile size and row tile size
    float softmax_scale,         // 1/sqrt(d)
    float *__restrict__ l,       // storage temp for row \sum exp(S)
    float *__restrict__ m,       // storage temp for row \max S
    float *__restrict__ O)       // output attention
{
  auto batch = blockIdx.x;
  auto head = blockIdx.y;
  auto numHeads = gridDim.y;

  auto qkv_offset = (batch * numHeads * N * d) + (head * N * d);
  auto lm_offset = (batch * numHeads * N) + (head * N);

  extern __shared__ float sram[];
  float *Qi = sram;                    // size = Br x d
  float *Kj = &sram[Br * d];           // size = Bc x d
  float *Vj = &sram[d * (Bc + Br)];    // size = Bc x d
  float *S = &sram[d * (Br + 2 * Bc)]; // size = Br x Bc

  auto const tx = threadIdx.x;
  auto const ty = threadIdx.y;
  auto const warp = ty;
  auto const numWarps = blockDim.y;

  auto const iStart = blockIdx.z * Br;
  auto const iEnd = min(iStart + Br, N);
  auto const Brc = min(Br, N - iStart);

  // set l and m to default values
  for (int i = iStart + ty * blockDim.x + tx; i < iEnd; i += warpSize * numWarps) {
    l[lm_offset + i] = 0.f;
    m[lm_offset + i] = -INFINITY;
  }

  // Load Q tile
  for (int ii = warp; ii < Brc; ii += numWarps) {
    auto i = iStart + ii;
    for (int k = tx; k < d; k += blockDim.x) {
      Qi[ii * d + k] = Q[qkv_offset + i * d + k];
    }
  }

  for (int jStart = 0; jStart < N; jStart += Bc) { // loop j tiles
    // Potentially cropped Bc in the last tile
    auto Bcc = min(Bc, N - jStart);

    // load Kj, Vj
    for (int k = tx; k < d; k += blockDim.x) {
      auto jj = ty;
      auto j = jStart + jj;
      Kj[jj * d + k] = (j < N) ? K[qkv_offset + j * d + k] : 0.f;
      Vj[jj * d + k] = (j < N) ? V[qkv_offset + j * d + k] : 0.f;
    }
    __syncthreads();

    // Compute Sij and row_max
    for (int ii = warp; ii < Brc; ii += numWarps) {
      auto i = iStart + ii;

      float row_m = -INFINITY;
      for (int jj = tx; jj < Bcc; jj += blockDim.x) {
        float Sij = 0.f;
        for (int k = 0; k < d; k++) {
          Sij += Qi[ii * d + k] * Kj[jj * d + k];
        }
        Sij *= softmax_scale;
        S[Bc * ii + jj] = Sij;
        row_m = common::float_max(row_m, Sij);
      }
      row_m = common::warpReduce<common::float_max>(row_m);

      float row_l = 0.f;
      for (int jj = tx; jj < Bcc; jj += blockDim.x) {
        float Sij = __expf(S[Bc * ii + jj] - row_m);
        S[Bc * ii + jj] = Sij;
        row_l += Sij;
      }
      row_l = common::warpReduce<common::float_add>(row_l);

      float row_m_prev = m[lm_offset + i];
      float row_l_prev = l[lm_offset + i];
      float row_m_new = common::float_max(row_m_prev, row_m);
      float row_l_new = __expf(row_m_prev - row_m_new) * row_l_prev +
                        __expf(row_m - row_m_new) * row_l;

      // Product Oik = Pin * Vnk
      // O[Br,d] = S[Br, Bc] * V[Bc, d]
      for (int k = tx; k < d; k += blockDim.x) {
        float PinVnk = 0.f;
        for (int n = 0; n < Bc; n++) {
          PinVnk += S[Bc * ii + n] * Vj[n * d + k];
        }
        O[qkv_offset + i * d + k] =
            ((row_l_prev * __expf(row_m_prev - row_m_new) *
              O[qkv_offset + i * d + k]) +
             (__expf(row_m - row_m_new) * PinVnk)) /
            row_l_new;

        // save new l and m
        if (tx == 0) {
          m[lm_offset + i] = row_m_new;
          l[lm_offset + i] = row_l_new;
        }
      }
    }

    __syncthreads();
  }
}

}
