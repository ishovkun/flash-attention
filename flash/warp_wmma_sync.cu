#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include "cuda_constants.hpp"
#include "common.hpp"

namespace flash {

using namespace nvcuda;

__global__ void warp_wmma_sync(const float *Q, const float *K, const float *V,
                               const int N, const int d, const int Tc,
                               const int Tr, const int Bc, const int Br,
                               const float softmax_scale, float *l, float *m,
                               float *O) {
  int tx = threadIdx.x;
  int batch = blockIdx.x;
  int head = blockIdx.y; // batch and head index

  // Offset into Q,K,V,O,l,m - different for each batch and head
  int qkv_offset = (batch * gridDim.y * N * d) + (head * N * d);     // gridDim.y = nh
  int lm_offset = (batch * gridDim.y * N) + (head * N); // offset for l and m

  // padded dimension d for wmma alignment
  auto dp = common::nextMultiple(d, constants::WMMA_N);

  // Define SRAM for Q,K,V,S
  extern __shared__ float sram[];
  int tile_size = Bc * d; // size of Qi, Kj, Vj
  float *Qi = sram;
  float *Kj = &sram[Br * dp];
  float *Vj = &sram[dp * (Bc + Br)];
  float *Sij = &sram[dp*(Br + 2*Bc)];

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
    for (int jj = 0; jj < Bc; jj++ ) {
      for (auto k = tx; k < dp; k += warpSize) {
        auto inBounds = jj < Bcc && k < d;
        Kj[jj*dp + k] = inBounds ? K[qkv_offset + j*tile_size + jj*d + k] : 0.f;
        Vj[jj*dp + k] = inBounds ? V[qkv_offset + j*tile_size + jj*d + k] : 0.f;
      }
    }
    __syncthreads();

    for (int i = 0; i < Tr; i++) {

      // Load Qi to SRAM
      for (int ii = 0; ii < Br; ii++) {
        for (auto k = tx; k < dp; k += warpSize) {
          auto inBounds = i*Br + ii < N && k < d;
          Qi[ii*dp + k] = inBounds ? Q[qkv_offset + (tile_size * i) + ii*d + k] : 0.f;
        }
      }
      __syncthreads();

      // Load l and m to registers
      float row_m_prev = -INFINITY, row_l_prev = 0.f;
      if (tx < Br) {
        row_m_prev = m[lm_offset + (Br * i) + tx];
        row_l_prev = l[lm_offset + (Br * i) + tx];
      }

      // S = QK^T - tensor cores going brrr
      constants::fragA_t q_frag;    // (16x8) WMMA_M x WMMA_K, row_major
      constants::fragB_cm_t k_frag; // (8x16) WMMA_K x WMMA_N, col_major
      constants::fragC_t s_frag;
      fill_fragment(s_frag, 0.0f);

      for (int k = 0; k < d; k += constants::WMMA_K) {
        wmma::load_matrix_sync(q_frag, Qi + k, dp);
        wmma::load_matrix_sync(k_frag, Kj + k, dp);
        // S_frag += q_frag * k_frag
        wmma::mma_sync(s_frag, q_frag, k_frag, s_frag);
      }
      wmma::store_matrix_sync(Sij, s_frag, constants::WMMA_M,
                              wmma::mem_row_major);

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
          Sij[(Bc * tx) + x] = __expf(Sij[(Bc * tx) + x] - row_m);
          row_l += Sij[(Bc * tx) + x];
        }
      }

      // PV = Pij * Vj - tensor cores going brrr again
      // using namespace wmma;
      constants::fragA_t p_frag;
      constants::fragB_rm_t v_frag;
      constants::fragC_t pv_frag;

      for (int x = 0; x < d; x += constants::WMMA_M) {
        wmma::fill_fragment(pv_frag, 0.0f);
        for (int k = 0; k < Bc; k += constants::WMMA_K) {
          wmma::load_matrix_sync(p_frag, Sij + k, Bc);
          wmma::load_matrix_sync(v_frag, Vj + x + (k * dp), dp);
          wmma::mma_sync(pv_frag, p_frag, v_frag, pv_frag);
        }
        // store it in unused Qi
        wmma::store_matrix_sync(Qi + x, pv_frag, dp, wmma::mem_row_major);
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
               (__expf(row_m - row_m_new) * Qi[(tx * dp) + x]));
        }
        m[lm_offset + (Br * i) + tx] = row_m_new;
        l[lm_offset + (Br * i) + tx] = row_l_new;
      }
    }
    __syncthreads();
  }
}

} // namespace flash
