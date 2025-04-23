#include "cuda_constants.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>

namespace flash {

using namespace nvcuda;
using namespace flash::constants;

__global__ void warp_wmma_sync(const float *Q, const float *K, const float *V,
                               const int N, const int d, const int Tc,
                               const int Tr, const int Bc, const int Br,
                               const float softmax_scale, float *l, float *m,
                               float *O) {
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int by = blockIdx.y; // batch and head index

  // Offset into Q,K,V,O,l,m - different for each batch and head
  int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d); // gridDim.y = nh
  int lm_offset = (bx * gridDim.y * N) + (by * N); // offset for l and m

  // Define SRAM for Q,K,V,S
  extern __shared__ float sram[];
  int tile_size = Bc * d; // size of Qi, Kj, Vj
  float *Qi = sram;
  float *Kj = &sram[tile_size];
  float *Vj = &sram[tile_size * 2];
  float *Sij = &sram[tile_size * 3];

  // Initialize l and m
  for (int x = 0; x < N; x += warpSize) {
    if (x + tx < N) {
      l[lm_offset + tx + x] = 0;
      m[lm_offset + tx + x] = -INFINITY;
    }
  }

  for (int j = 0; j < Tc; j++) {

    int const Bcc = min(Bc, N - j * Bc);

    // Load Kj, Vj to SRAM
    for (int x = 0; x < tile_size; x += warpSize) {
      auto jj = x / d;
      auto inBounds = x + tx < tile_size && j * Bc + jj < N;
        Kj[x + tx] = inBounds ? K[qkv_offset + (tile_size * j) + x + tx] : 0.f;
        Vj[x + tx] = inBounds ? V[qkv_offset + (tile_size * j) + x + tx] : 0.f;
      // if (x + tx < tile_size && j * Bc + jj < N) {
      //   // TF32 conversion for WMMA
      //   Kj[x + tx] = K[qkv_offset + (tile_size * j) + x + tx];
      //   Vj[x + tx] = V[qkv_offset + (tile_size * j) + x + tx];
      // }
    }
    __syncthreads();

    for (int i = 0; i < Tr; i++) {

      // Load Qi to SRAM
      for (int x = 0; x < tile_size; x += warpSize) {
        auto ii = x / d;
        auto inBounds = x + tx < tile_size && i * Br + ii < N;
        Qi[x + tx] = inBounds ? Q[qkv_offset + (tile_size * i) + x + tx] : 0.f;
        // if (x + tx < tile_size && i * Br + ii < N)
          // Qi[x + tx] = Q[qkv_offset + (tile_size * i) + x + tx];
      }
      __syncthreads();

      // Load l and m to registers
      float row_m_prev = -INFINITY, row_l_prev = 0.f;
      if (tx < Br) {
        row_m_prev = m[lm_offset + (Br * i) + tx];
        row_l_prev = l[lm_offset + (Br * i) + tx];
      }

      // S = QK^T - tensor cores going brrr
      using namespace wmma;
      fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, precision::tf32, row_major> q_frag;
      // note col_major for transpose
      fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, precision::tf32, col_major> k_frag;
      fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> s_frag;
      fill_fragment(s_frag, 0.0f);

      for (int k = 0; k < d; k += WMMA_K) {
        load_matrix_sync(q_frag, Qi + k, d);
        load_matrix_sync(k_frag, Kj + k, d);
        mma_sync(s_frag, q_frag, k_frag, s_frag);
      }
      store_matrix_sync(Sij, s_frag, WMMA_M, mem_row_major);

      float row_m = -INFINITY;
      float row_l = 0;
      if (tx < Br && Bc * i + tx < N) {
        // Softmax scaling, row_m = rowmax(S)
        for (int x = 0; x < Bcc; x++) {
          Sij[(Bc * tx) + x] *= softmax_scale;
          row_m = max(row_m, Sij[(Bc * tx) + x]);
        }

        // P = exp(S - row_m), row_l = rowsum(P)
        for (int x = 0; x < Bcc; x++) {
          Sij[(Bc * tx) + x] = (x < Bcc) ? __expf(Sij[(Bc * tx) + x] - row_m) : 0.0f;
          row_l += Sij[(Bc * tx) + x];
        }
      }

      // PV = Pij * Vj - tensor cores going brrr again
      fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, precision::tf32, row_major> p_frag;
      fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, precision::tf32, row_major> v_frag;
      fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> pv_frag;

      for (int x = 0; x < d; x += WMMA_M) {
        wmma::fill_fragment(pv_frag, 0.0f);
        for (int k = 0; k < Br; k += WMMA_K) {
          wmma::load_matrix_sync(p_frag, Sij + k, Bc);
          wmma::load_matrix_sync(v_frag, Vj + x + (k * d), d);
          wmma::mma_sync(pv_frag, p_frag, v_frag, pv_frag);
        }
        // store it in unused Qi
        wmma::store_matrix_sync(Qi + x, pv_frag, d, wmma::mem_row_major);
      }

      if (tx < Br && i * Br + tx < N) {
        // Compute new m and l
        float row_m_new = max(row_m_prev, row_m);
        float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) +
                          (__expf(row_m - row_m_new) * row_l);

        // Write O, l, m to HBM
        for (int x = 0; x < d; x++) {
          O[qkv_offset + (tile_size * i) + (tx * d) + x] =
              (1 / row_l_new) *
              ((row_l_prev * __expf(row_m_prev - row_m_new) *
                O[qkv_offset + (tile_size * i) + (tx * d) + x]) +
               (__expf(row_m - row_m_new) * Qi[(tx * d) + x]));
        }
        m[lm_offset + (Br * i) + tx] = row_m_new;
        l[lm_offset + (Br * i) + tx] = row_l_new;
      }
    }
    __syncthreads();
  }
}

} // namespace flash
