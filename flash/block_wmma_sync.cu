#include "common.hpp"
#include "cuda_constants.hpp"
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>

namespace flash {

using namespace nvcuda;

__global__ void
block_wmma_sync(float const *__restrict__ Q, // query vector
                float const *__restrict__ K, // key vector
                float const *__restrict__ V, // value vector
                int N,                       // sequence length
                int d,                       // head_dim
                int Bc, int Br,        // column tile size and row tile size
                float softmax_scale,   // 1/sqrt(d)
                float *__restrict__ l, // storage temp for row \sum exp(S)
                float *__restrict__ m, // storage temp for row \max S
                float *__restrict__ O) // output attention
{
  int bx = blockIdx.x; // batch index
  int by = blockIdx.y; // head index

  int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d); // gridDim.y = nh
  int lm_offset = (bx * gridDim.y * N) + (by * N); // offset for l and m

  extern __shared__ float sram[];
  float *_Q = sram;                     // size = Br x d
  float *_K = &sram[Br * d];            // size = Bc x d
  float *_V = &sram[d * (Bc + Br)];     // size = Bc x d
  float *_S = &sram[d * (Br + 2 * Bc)]; // size = Br x Bc

  auto tx = threadIdx.x;
  auto ty = threadIdx.y;

  for (int jStart = 0; jStart < N; jStart += Bc) { // loop j tiles
    // Potentially cropped Bc in the last tile
    auto Bcc = min(Bc, N - jStart);
    // load Kj, Vj
    for (int k = tx; k < d; k += blockDim.x) {
      auto jj = ty;
      auto j = jStart + jj;
      _K[jj * d + k] = (j < N) ? K[qkv_offset + j * d + k] : 0.f;
      _V[jj * d + k] = (j < N) ? V[qkv_offset + j * d + k] : 0.f;
    }

    for (int iStart = 0; iStart < N; iStart += Br) { // loop i tiles

      // Load Qi
      for (int k = tx; k < d; k += blockDim.x) {
        auto ii = ty;
        auto i = iStart + ii;
        _Q[ii * d + k] = (i < N) ? Q[qkv_offset + i * d + k] : 0.f;
      }
      __syncthreads();

      // Compute Sij
      constants::fragA_t q_frag;    // (16x8) WMMA_M x WMMA_K, row_major
      constants::fragB_cm_t k_frag; // (8x16) WMMA_K x WMMA_N, col_major
      constants::fragC_t s_frag;

      int warp = ty;
      // Split Sij into numWarpsI x numWarpsJ
      auto numSubtilesI = Br / constants::WMMA_M;
      auto numSubtilesJ = Bc / constants::WMMA_N;
      auto numSubtiles = numSubtilesI * numSubtilesJ;
      auto numWarps = blockDim.y;
      for (int subTile = warp; subTile < numSubtiles; subTile += numWarps) {
        fill_fragment(s_frag, 0.f);
        auto subtileI = subTile / numSubtilesJ;
        auto subtileJ = subTile % numSubtilesJ;
        auto ii = subtileI * constants::WMMA_M;
        auto jj = subtileJ * constants::WMMA_N;
        for (int k = 0; k < d; k += constants::WMMA_K) {
          wmma::load_matrix_sync(q_frag, &_Q[ii * d + k], d);
          wmma::load_matrix_sync(k_frag, &_K[jj * d + k], d);
          mma_sync(s_frag, q_frag, k_frag, s_frag);
        }
        wmma::store_matrix_sync(&_S[ii * Bc + jj], s_frag, constants::WMMA_M,
                                wmma::mem_row_major);
      }
      __syncthreads();

      // compute row max
      float row_m = -INFINITY;
      for (int jj = tx; jj < Bcc; jj += blockDim.x) {
        auto ii = ty;
        _S[Bc * ii + jj] *= softmax_scale;
        row_m = max(row_m, _S[Bc * ii + jj]);
      }
      row_m = common::warpReduce<common::float_max>(row_m);

      // compute row sum and P = exp(Sij)
      float row_l = 0.f;
      for (int jj = tx; jj < Bcc; jj += blockDim.x) {
        auto ii = ty;
        float Sij = __expf(_S[Bc * ii + jj] - row_m);
        _S[Bc * ii + jj] = Sij;
        row_l += Sij;
      }
      row_l = common::warpReduce<common::float_add>(row_l);

      auto i = iStart + ty;
      float row_m_prev = (i < N) ? m[lm_offset + i] : -INFINITY;
      float row_l_prev = (i < N) ? l[lm_offset + i] : 0.f;
      float row_m_new = common::float_max(row_m_prev, row_m);
      float row_l_new = __expf(row_m_prev - row_m_new) * row_l_prev +
                        __expf(row_m - row_m_new) * row_l;
      __syncthreads();

      // compute pv[i,k] = P[i,j] * V[k,j]
      constants::fragA_t p_frag;
      constants::fragB_rm_t v_frag;
      constants::fragC_t pv_frag;

      auto numSubtilesK = d / constants::WMMA_N;
      auto numSubtilesTotal = numSubtilesI * numSubtilesJ;
      auto *_PV = _Q;
      for (int subTile = warp; subTile < numSubtilesTotal;
           subTile += numWarps) {
        fill_fragment(p_frag, 0.0f);
        auto subtileI = subTile / numSubtilesK;
        auto subtileK = subTile % numSubtilesK;
        auto ii = subtileI * constants::WMMA_M;
        auto kk = subtileK * constants::WMMA_N;
        for (int jj = 0; jj < Bcc; jj += constants::WMMA_N) {
          wmma::load_matrix_sync(p_frag, &_S[ii * Bc + jj], Bc);
          wmma::load_matrix_sync(v_frag, &_V[jj * d + kk], d);
          wmma::mma_sync(pv_frag, p_frag, v_frag, pv_frag);
        }
        wmma::store_matrix_sync(&_PV[ii * d + kk], pv_frag, d,
                                wmma::mem_row_major);
      }
      __syncthreads();

      // Write O,l, and m to global memory
      auto ii = ty;
      i = iStart + ii;
      if (i < N) {
        for (int k = tx; k < d; k += blockDim.x) {
          O[qkv_offset + i * d + k] =
              ((row_l_prev * __expf(row_m_prev - row_m_new) *
                O[qkv_offset + i * d + k]) +
               (__expf(row_m - row_m_new) * _PV[ii * d + k])) /
              row_l_new;
        }
        if (tx == 0) {
          m[lm_offset + i] = row_m_new;
          l[lm_offset + i] = row_l_new;
        }
      }
      __syncthreads();
    }
  }
}

} // end namespace flash
