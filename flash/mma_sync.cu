#include "common.hpp"
#include "mma.hpp"
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <mma.h>

namespace flash {

__global__ void forward_kernel_mma_sync(
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
  int batch = blockIdx.x;
  int head = blockIdx.y;
  int numHeads = gridDim.y;

  int qkv_offset = (batch * numHeads + head) * N * d;
  int lm_offset = (batch * numHeads + head) * N;

  // padded dimension d for wmma alignment
  auto dp = common::nextMultiple(d, mma::Tile::N);

  // analysis give Br = 48
  // local mcur size should be Br / warpsPerBlock
  float mcur[48];

  // shared memory for tiles
  extern __shared__ float sram[];
  float *_Q = sram;                      // size = Br x d
  float *_K = &sram[Br * dp];            // size = Bc x d
  float *_V = &sram[dp * (Bc + Br)];     // size = Bc x d
  float *_S = &sram[dp * (Br + 2 * Bc)]; // size = Br x Bc

  auto const tx = threadIdx.x;
  auto const ty = threadIdx.y;
  auto const warp = ty;
  auto const numWarps = blockDim.y;

  auto const iStart = blockIdx.z * Br;
  auto const iEnd = min(iStart + Br, N);
  auto const Brc = min(Br, N - iStart);

  // set l and m to default values
  for (int i = iStart + warp * warpSize + tx; i < iEnd; i += warpSize * numWarps) {
    l[lm_offset + i] = 0.f;
    m[lm_offset + i] = -INFINITY;
  }

  // Split Sij into numWarpsI x numWarpsJ
  // auto rowsPerWarp = Br / numWarps;


  // Load Q tile
  for (int ii = warp; ii < Brc; ii += numWarps) {
    auto i = iStart + ii;
    for (int k = tx; k < dp; k += blockDim.x) {
      auto inBounds = k < d;
      _Q[ii * dp + k] = inBounds ? Q[qkv_offset + i * d + k] : 0.f;
    }
  }

  for (int jStart = 0; jStart < N; jStart += Bc) { // loop j tiles
    // Potentially cropped Bc in the last tile
    auto Bcc = min(Bc, N - jStart);

    // load Kj, Vj
    for (int jj = warp; jj < Bc; jj += numWarps) {
      auto j = jStart + jj;
      for (int k = tx; k < dp; k += blockDim.x) {
        auto inBounds = jj < Bcc && k < d;
        _K[jj * dp + k] = inBounds ? K[qkv_offset + j * d + k] : 0.f;
        _V[jj * dp + k] = inBounds ? V[qkv_offset + j * d + k] : 0.f;
      }
    }

    __syncthreads();

    // Compute Sij
    mma::FragmentA q_frag;
    mma::FragmentB<mma::Layout::col_major> k_frag;
    mma::FragmentAccumulator s_frag;

    auto numSubtilesI = common::ceil_div(Br, mma::Tile::M);
    auto numSubtilesJ = common::ceil_div(Bc, mma::Tile::N);
    auto numSubtilesQK = numSubtilesI * numSubtilesJ;
    for (int subTile = warp; subTile < numSubtilesQK; subTile += numWarps) {
      mma::fill_fragment(s_frag, 0.f);
      auto subtileI = subTile / numSubtilesJ;
      auto subtileJ = subTile % numSubtilesJ;
      auto ii = subtileI * mma::Tile::M;
      auto jj = subtileJ * mma::Tile::N;
      for (int k = 0; k < d; k += mma::Tile::K) {
        mma::load_matrix_sync(q_frag, &_Q[ii * dp + k], dp); // 16 x 8
        mma::load_matrix_sync(k_frag, &_K[jj * dp + k], dp); // 8 x 8
        mma::mma_sync(s_frag, q_frag, k_frag, s_frag);
      }
      // apply scaling
      for (int t = 0; t < s_frag.size; t++)
        s_frag.reg[t] *= softmax_scale;
      mma::store_matrix_sync(&_S[ii * Bc + jj], s_frag, Bc);
    }
    __syncthreads();

    // P = exp(S)
    for (int ii = warp; ii < Br; ii += numWarps) {
      float row_m = -INFINITY;
      for (int jj = tx; jj < Bcc; jj += blockDim.x) {
        row_m = common::float_max(row_m, _S[Bc * ii + jj]);
      }
      row_m = common::warpReduce<common::float_max>(row_m);
      mcur[ii] = row_m;
      for (int jj = tx; jj < Bc; jj += blockDim.x) {
        float Pij = __expf(_S[Bc * ii + jj] - row_m);
        _S[Bc * ii + jj] = (ii < Brc && jj < Bcc) ? Pij : 0.f;
      }
    }
    __syncthreads();

    float *_PV = _K; // reuse _K to store PV
    // for (int ii = warp; ii < Brc; ii += numWarps) {
    //   for (int k = tx; k < d; k += blockDim.x) {
    //     float PinVnk = 0.f;
    //     for (int jj = 0; jj < Bc; jj++) {
    //       PinVnk += _S[Bc * ii + jj] * _V[jj * dp + k];
    //     }
    //     _PV[ii*dp + k] = PinVnk;
    //   }
    // }

    mma::FragmentA p_frag;
    mma::FragmentB<mma::Layout::row_major> v_frag;
    mma::FragmentAccumulator pv_frag;

    auto numSubtilesK = dp / mma::Tile::N;
    auto numSubtilesPV = numSubtilesI * numSubtilesK;
    for (int subTile = warp; subTile < numSubtilesPV; subTile += numWarps) {
      fill_fragment(pv_frag, 0.f);
      auto subtileI = subTile / numSubtilesK;
      auto subtileK = subTile % numSubtilesK;
      auto ii = subtileI * mma::Tile::M;
      auto k = subtileK * mma::Tile::N;
      for (int jj = 0; jj < Bc; jj += mma::Tile::K) {
        mma::load_matrix_sync(p_frag, &_S[ii * Bc + jj], Bc); // P: Br x Bc
        mma::load_matrix_sync(v_frag, &_V[jj * dp + k], dp);  // V: Bc x d
        mma::mma_sync(pv_frag, p_frag, v_frag, pv_frag);
      }
      mma::store_matrix_sync(&_PV[ii * dp + k], pv_frag, dp);
    }
    __syncthreads();

    // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tx == 0 &&
    //     ty == 0 && jStart == 0) {
    //   {
    //     auto ii = 0;
    //     for (int k = 0; k < mma::Tile::K; k++) {
    //       float PinVnk = 0.f;
    //       for (int jj = 0; jj < Bc; jj++) {
    //         PinVnk += _S[ii * Bc + jj] * _V[jj * dp + k];
    //       }
    //       auto diff = _PV[ii*dp + k] - PinVnk;
    //       printf("PV[%d %d] = %f %f diff %f\n", ii, k, _PV[ii*dp + k], PinVnk, diff);

    //       {
    //         // Frag B
    //       }
    //     }
    //   }
    // }

    // Write O,l, and m to global memory
    for (int ii = warp; ii < Brc; ii += numWarps) {
      auto i = iStart + ii;
      auto row_m = mcur[ii];

      // compute row_l
      float row_l = 0.f;
      for (int jj = tx; jj < Bcc; jj += warpSize) {
        row_l += _S[Bc * ii + jj];
      }
      row_l = common::warpReduce<common::float_add>(row_l);

      float row_m_prev = m[lm_offset + i];
      float row_l_prev = l[lm_offset + i];
      float row_m_new = common::float_max(row_m_prev, row_m);
      float row_l_new = __expf(row_m_prev - row_m_new) * row_l_prev +
                        __expf(row_m - row_m_new) * row_l;

      // compute O
      for (int k = tx; k < d; k += blockDim.x) {
        O[qkv_offset + i * d + k] =
            ((row_l_prev * __expf(row_m_prev - row_m_new) *
              O[qkv_offset + i * d + k]) +
             (__expf(row_m - row_m_new) * _PV[ii * dp + k])) /
            row_l_new;
      }
      // save row max and row sum
      if (tx == 0) {
        m[lm_offset + i] = row_m_new;
        l[lm_offset + i] = row_l_new;
      }
    }
    __syncthreads();
  }
}

} // end namespace flash
