#include "common.hpp"
#include "cuda_constants.hpp"
#include <cmath>
#include <cstdio>
#include <cuda.h>
#include <cuda/pipeline>
#include <cuda_runtime.h>
#include <iostream>
#include <mma.h>

namespace flash {

using namespace nvcuda;

__global__ void
block_wmma_async(float const *__restrict__ Q, // query vector
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
  int batch = blockIdx.x;
  int head = blockIdx.y;
  int numHeads = gridDim.y;

  int qkv_offset = (batch * numHeads + head) * N * d;
  int lm_offset = (batch * numHeads + head) * N;

  // padded dimension d for wmma alignment
  auto dp = common::nextMultiple(d, constants::WMMA_N);

  // shared memory for tiles
  extern __shared__ float sram[];
  float *_Q = sram;                             // size = Br x d
  float *_K = &sram[Br * dp];                   // size = Bc x d
  float *_V = &sram[dp * (Bc + Br)];            // size = Bc x d
  float *_S = &sram[dp * (Br + 2 * Bc)];        // size = Br x Bc
  float *_m = &sram[dp * (Br + 2 * Bc) + Br*Bc]; // size = max(Br, Bc)

  auto tx = threadIdx.x;
  auto ty = threadIdx.y;

  // set l and m to default values
  for (int i = ty * blockDim.x + tx; i < N; i += blockDim.x * blockDim.y) {
    l[lm_offset + i] = 0.f;
    m[lm_offset + i] = -INFINITY;
  }

  auto pipe = cuda::make_pipeline();
  auto aligned_float = cuda::aligned_size_t<alignof(float)>(sizeof(float));

  auto numWarps = blockDim.y;
  int warp = ty;

  for (int jStart = 0; jStart < N; jStart += Bc) { // loop j tiles
    // Potentially cropped Bc in the last tile
    auto Bcc = min(Bc, N - jStart);

    // load Kj, Vj
    pipe.producer_acquire();
    for (int jj = warp; jj < Bc; jj += numWarps) {
      for (int k = tx; k < dp; k += blockDim.x) {
        auto j = jStart + jj;
        auto inBounds = jj < Bcc && k < d;
        if (inBounds) {
          cuda::memcpy_async(&_K[jj * dp + k], &K[qkv_offset + j * d + k],
                             aligned_float, pipe);
          cuda::memcpy_async(&_V[jj * dp + k], &V[qkv_offset + j * d + k],
                             aligned_float, pipe);
        } else {
          _K[jj * dp + k] = 0.f;
          _V[jj * dp + k] = 0.f;
        }
      }
    }
    pipe.producer_commit();

    for (int iStart = 0; iStart < N; iStart += Br) { // loop i tiles
      auto Brc = min(Bc, N - iStart);

      // Load Qi
      pipe.producer_acquire();
      for (int ii = warp; ii < Br; ii += numWarps) {
        auto i = iStart + ii;
        for (int k = tx; k < dp; k += constants::warpSize) {
          auto inBounds = i < N && k < d;
          // _Q[ii * dp + k] = inBounds ? Q[qkv_offset + i * d + k] : 0.f;
          if (inBounds) {
            cuda::memcpy_async(&_Q[ii * dp + k], &Q[qkv_offset + i * d + k],
                               aligned_float, pipe);
          }
          else _Q[ii * dp + k] = 0.f;
        }
      }
      pipe.producer_commit();
      cuda::pipeline_consumer_wait_prior<0>(pipe);
      __syncthreads();

      // Compute Sij
      constants::fragA_t q_frag;    // (16x8) WMMA_M x WMMA_K, row_major
      constants::fragB_cm_t k_frag; // (8x16) WMMA_K x WMMA_N, col_major
      constants::fragC_t s_frag;    // (16x16) WMMA_M x WMMA_N, row_major

      // Split Sij into numWarpsI x numWarpsJ
      auto numSubtilesI = common::ceil_div(Br, constants::WMMA_M);
      auto numSubtilesJ = common::ceil_div(Bc, constants::WMMA_N);
      auto numSubtilesQK = numSubtilesI * numSubtilesJ;

      for (int subTile = warp; subTile < numSubtilesQK; subTile += numWarps) {
        fill_fragment(s_frag, 0.f);
        auto subtileI = subTile / numSubtilesJ;
        auto subtileJ = subTile % numSubtilesJ;
        auto ii = subtileI * constants::WMMA_M;
        auto jj = subtileJ * constants::WMMA_N;
        for (int k = 0; k < d; k += constants::WMMA_K) {
          wmma::load_matrix_sync(q_frag, &_Q[ii * dp + k], dp);
          wmma::load_matrix_sync(k_frag, &_K[jj * dp + k], dp);

          // Round to the nearest tf32
          for (int t = 0; t < q_frag.num_elements; t++)
            q_frag.x[t] = wmma::__float_to_tf32(q_frag.x[t]);
          for (int t = 0; t < k_frag.num_elements; t++)
            k_frag.x[t] = wmma::__float_to_tf32(k_frag.x[t]);

          wmma::mma_sync(s_frag, q_frag, k_frag, s_frag);
        }
        // apply scaling
        for (int t = 0; t < s_frag.num_elements; t++)
          s_frag.x[t] *= softmax_scale;

        wmma::store_matrix_sync(&_S[ii * Bc + jj], s_frag, Bc,
                                wmma::mem_row_major);
      }
      __syncthreads();

      for (int ii = warp; ii < Br; ii += numWarps) {
        float row_m = -INFINITY;
        // compute row max
        for (int jj = tx; jj < Bcc; jj += blockDim.x) {
          row_m = common::float_max(row_m, _S[Bc * ii + jj]);
        }
        row_m = common::warpReduce<common::float_max>(row_m);
        if (tx == 0) _m[ii] = row_m;

        // compute row sum and P = exp(Sij)
        // for (int jj = tx; jj < Bcc; jj += blockDim.x) {
        //   float Pij = __expf(_S[Bc * ii + jj] - row_m);
        //   _S[Bc * ii + jj] = (iStart + ii < N) ? Pij : 0.f;
        // }
        for (int jj = tx; jj < Bc; jj += blockDim.x) {
          float Pij = __expf(_S[Bc * ii + jj] - row_m);
          _S[Bc * ii + jj] = (iStart + ii < N && jj < Bcc) ? Pij : 0.f;
        }
      }
      __syncthreads();

      // compute pv[i,k] = P[i,j] * V[k,j]
      float *_PV = _Q; // reuse _Q to store PV

      constants::fragA_t p_frag;
      constants::fragB_rm_t v_frag;
      constants::fragC_t pv_frag;

      auto numSubtilesK = dp / constants::WMMA_N;
      auto numSubtilesPV = numSubtilesI * numSubtilesK;
      for (int subTile = warp; subTile < numSubtilesPV; subTile += numWarps) {
        fill_fragment(pv_frag, 0.f);
        auto subtileI = subTile / numSubtilesK;
        auto subtileK = subTile % numSubtilesK;
        auto ii = subtileI * constants::WMMA_M;
        auto k = subtileK * constants::WMMA_N;
        for (int jj = 0; jj < Bc; jj += constants::WMMA_K) {
          wmma::load_matrix_sync(p_frag, &_S[ii * Bc + jj], Bc); // P: Br x Bc
          wmma::load_matrix_sync(v_frag, &_V[jj * dp + k], dp);  // V: Bc x d
          for (int t = 0; t < p_frag.num_elements; t++)
            p_frag.x[t] = wmma::__float_to_tf32(p_frag.x[t]);
          for (int t = 0; t < v_frag.num_elements; t++)
            v_frag.x[t] = wmma::__float_to_tf32(v_frag.x[t]);
          wmma::mma_sync(pv_frag, p_frag, v_frag, pv_frag);
        }
        wmma::store_matrix_sync(&_PV[ii * dp + k], pv_frag, dp,
                                wmma::mem_row_major);
      }
      __syncthreads();

      // Write O,l, and m to global memory
      for (int ii = warp; ii < Brc; ii += numWarps) {
        auto i = iStart + ii;
        auto row_m = _m[ii];

        // compute row_l
        float row_l = 0.f;
        for (int jj = tx; jj < Bcc; jj += blockDim.x) {
          row_l += _S[Bc * ii + jj];
        }
        row_l = common::warpReduce<common::float_add>(row_l);

        float row_m_prev = m[lm_offset + i];
        float row_l_prev = l[lm_offset + i];
        float row_m_new = common::float_max(row_m_prev, row_m);
        float row_l_new = __expf(row_m_prev - row_m_new) * row_l_prev +
                          __expf(row_m - row_m_new) * row_l;

        if (i == 64) {
          printf("I am here %d %d\n", tx, ty);
        }

        // compute O
        for (int k = tx; k < d; k += blockDim.x) {
          O[qkv_offset + i * d + k] =
              ((row_l_prev * __expf(row_m_prev - row_m_new) *
                O[qkv_offset + i * d + k]) +
               (__expf(row_m - row_m_new) * _PV[ii * dp + k])) / row_l_new;
        }
        // save row max and row sum
        if (tx == 0) {
          m[lm_offset + i] = row_m_new;
          l[lm_offset + i] = row_l_new;
        }
      }
      pipe.consumer_release(); // release the consumed stage
      __syncthreads();
    }
  }
}

} // namespace flash
