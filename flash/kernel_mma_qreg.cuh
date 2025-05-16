#pragma once
#include "common.hpp"
#include "mma.hpp"
#include <cassert>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
// #include <iostream>
#include <mma.h>

namespace flash {

/*
 * In this version, Q is distributed between the registers
 */
template <uint32_t Br, uint32_t Bc, uint32_t numWarps, uint32_t maxHeadDim>
__global__ void
kernel_mma_qreg(float const *__restrict__ Q, // query vector
                float const *__restrict__ K, // key vector
                float const *__restrict__ V, // value vector
                int seq_length,              // sequence length
                int d,                       // head_dim
                float softmax_scale,         // 1/sqrt(d)
                float *__restrict__ l,       // storage temp for row \sum exp(S)
                float *__restrict__ m,       // storage temp for row \max S
                float *__restrict__ O)       // output attention
{
  static_assert(numWarps * mma::Tile::M >= Br);
  static_assert((numWarps * mma::Tile::M) % Br == 0,
                "Each subtile has the same number of warps");

  int batch = blockIdx.x;
  int head = blockIdx.y;
  int numHeads = gridDim.y;

  int qkv_offset = (batch * numHeads + head) * seq_length * d;
  int lm_offset = (batch * numHeads + head) * seq_length;

  // padded dimension d for wmma alignment
  uint32_t dp = maxHeadDim;
  assert(dp >= d);

  // shared memory for tiles
  extern __shared__ float sram[];

  // We load a Q tile into _Q, and then store Q tile in registers
  // then we load a K tile into _K (same location as Q, but maybe different size
  // Bc) and we load a V tile into _V
  float *_Q = sram;               // size = Br * dp
  float *_K = sram;               // size = Bc * dp
  float *_V = &sram[dp * Bc];     // size = Bc x dp
  float *_S = &sram[dp * max(2 * Bc, Br)]; // size = Br x Bc

  auto const tx = threadIdx.x;
  auto const ty = threadIdx.y;
  auto const warp = ty;

  auto const iStart = blockIdx.z * Br;
  auto const iEnd = min(iStart + Br, seq_length);
  auto const Brc = min(Br, seq_length - iStart);

  constexpr auto numSubtilesI = common::ceil_div(Br, mma::Tile::M);
  constexpr auto numSubtilesK = common::ceil_div(maxHeadDim, mma::Tile::K);
  constexpr auto numSubtilesJ = common::ceil_div(Bc, mma::Tile::N);

  auto const wy = warp % numSubtilesI;
  auto const wx = warp / numSubtilesI;
  constexpr auto numWarpsX = numWarps / numSubtilesI;

  const auto subtileI = wy;
  auto const kStart = wx * common::warpSize + tx;

  // set l and m to default values
  for (int i = iStart + warp * warpSize + tx; i < iEnd;
       i += warpSize * numWarps) {
    l[lm_offset + i] = 0.f;
    m[lm_offset + i] = -INFINITY;
  }

  // Load Q tile
  auto iiStart = subtileI * mma::Tile::M;
  auto iiEnd = min((subtileI + 1) * mma::Tile::M, Brc);
  for (uint32_t ii = iiStart; ii < iiEnd; ii++) {
    auto i = iStart + ii;
    for (uint32_t k = kStart; k < maxHeadDim; k += numWarpsX * warpSize) {
      auto inBounds = k < d;
      _Q[ii * dp + k] = inBounds ? Q[qkv_offset + i * d + k] : 0.f;
    }
  }

  if constexpr (numWarpsX > 1) {
    __syncthreads();
  }
  __syncthreads();

  // Load full Q into registers of every warp
  mma::FragmentA q_frag[numSubtilesK];
  for (uint32_t subtileK = 0; subtileK < numSubtilesK; subtileK++) {
    auto const k = subtileK * mma::Tile::K;
    mma::load_matrix_sync(q_frag[subtileK], _Q, iiStart, k, dp);
  }
  __syncthreads(); // this should now be here cause warps only pull rows relevant to their subtiles

  for (int jStart = 0; jStart < seq_length; jStart += Bc) { // loop j tiles
    // Potentially cropped Bc in the last tile
    auto Bcc = min(Bc, seq_length - jStart);

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
    mma::FragmentB<mma::Layout::col_major> k_frag;
    mma::FragmentAccumulator s_frag;

    constexpr auto subtilesJPerWarp = common::ceil_div(numSubtilesJ, numWarpsX);
    for (auto subtileJ = wx * subtilesJPerWarp; subtileJ < (wx + 1) * subtilesJPerWarp; subtileJ++) {
      auto const ii = subtileI * mma::Tile::M;
      auto const jj = subtileJ * mma::Tile::N;

      mma::fill_fragment(s_frag, 0.f);
      for (auto subtileK = 0; subtileK < numSubtilesK; subtileK++) {
        auto k = subtileK * mma::Tile::K;
        mma::load_matrix_sync(k_frag, _K, jj, k, dp);
        mma::mma_sync(s_frag, q_frag[subtileK], k_frag, s_frag);
      }
      for (int t = 0; t < s_frag.size; t++)
        s_frag.reg[t] *= softmax_scale;
      mma::store_matrix_sync(_S, ii, jj, Bc, s_frag);
    }

    if constexpr (numWarpsX > 1) {
      __syncthreads();
    }
    __syncthreads();

    constexpr uint32_t rowsPerWarp = mma::Tile::M / numWarpsX;
    iiStart = (subtileI * mma::Tile::M) + wx * rowsPerWarp;
    iiEnd = (subtileI * mma::Tile::M) + (wx + 1) * rowsPerWarp;
    float mcur[rowsPerWarp];
    float lcur[rowsPerWarp];
    for (auto ii = iiStart; ii < iiEnd; ii++) {
      float row_m = -INFINITY;
      for (auto jj = tx; jj < Bcc; jj += warpSize) {
        row_m = common::float_max(row_m, _S[Bc * ii + jj]);
      }
      row_m = common::warpReduce<common::float_max>(row_m);
      mcur[ii - iiStart] = row_m;
      float row_l = 0.f;
      for (int jj = tx; jj < Bc; jj += blockDim.x) {
        float Pij = __expf(_S[Bc * ii + jj] - row_m);
        Pij = (ii < Brc && jj < Bcc) ? Pij : 0.f;
        row_l += Pij;
        _S[Bc * ii + jj] = Pij;
      }
      row_l = common::warpReduce<common::float_add>(row_l);
      lcur[ii - iiStart] = row_l;
    }
    if constexpr (numWarpsX > 1) {
      __syncthreads();
    }
    __syncthreads(); // not needed

    mma::FragmentA p_frag;
    mma::FragmentB<mma::Layout::row_major> v_frag;
    mma::FragmentAccumulator pv_frag;

    constexpr auto subtilesKPerWarp = numSubtilesK / numWarpsX;
    iiStart = subtileI * mma::Tile::M;
    auto iiEnd = min((subtileI + 1) * mma::Tile::M, Brc);
    for (auto subtileK = wx * subtilesKPerWarp; subtileK < (wx + 1) * subtilesKPerWarp; subtileK++) {
      fill_fragment(pv_frag, 0.f);
      auto const frag_ii = subtileI * mma::Tile::M;
      auto const frag_k = subtileK * mma::Tile::N;
      for (uint32_t frag_jj = 0; frag_jj < Bc; frag_jj += mma::Tile::K) {
        mma::load_matrix_sync(p_frag, _S, frag_ii, frag_jj, Bc);
        mma::load_matrix_sync(v_frag, _V, frag_jj, frag_k, dp);
        mma::mma_sync(pv_frag, p_frag, v_frag, pv_frag);
      }
      // now we have pv_frag and we can directly write output
      // float rO[pv_frag.size];
      for (auto r = 0; r < pv_frag.size; r++) {
        auto lane = tx;
        auto group = lane >> 2;
        auto member = lane % 4;
        uint32_t fragRow = (r < 2) ? group : group + 8; // 0 to 15
        uint32_t fragCol = 2 * member + (r & 0x1); // 0 to 7
        auto ii = frag_ii + fragRow;
        auto i = iStart + ii;
        auto k = frag_k + fragCol;
        auto PVik = pv_frag.reg[r];

        // mcur should be stored in _Q
        auto const row_m = mcur[ii - iiStart];
        auto const row_l = lcur[ii - iiStart];

        if (i < seq_length && k < d) {
          float row_m_prev = m[lm_offset + i];
          float row_l_prev = l[lm_offset + i];
          float row_m_new = common::float_max(row_m_prev, row_m);
          float row_l_new = __expf(row_m_prev - row_m_new) * row_l_prev +
                            __expf(row_m - row_m_new) * row_l;
          O[qkv_offset + i * d + k] =
              ((row_l_prev * __expf(row_m_prev - row_m_new) *
                O[qkv_offset + i * d + k]) +
               (__expf(row_m - row_m_new) * PVik)) /
              row_l_new;
        }
      }
    }
    // store new l and m!!!!
    if (wx == 0 && tx == 0) {
      for (int ii = iiStart; ii < iiEnd; ii++) {
        auto i = iStart + ii;
        float row_m_prev = m[lm_offset + i];
        float row_l_prev = l[lm_offset + i];
        auto const row_m = mcur[ii - iiStart];
        auto const row_l = lcur[ii - iiStart];
        float row_m_new = common::float_max(row_m_prev, row_m);
        float row_l_new = __expf(row_m_prev - row_m_new) * row_l_prev +
                          __expf(row_m - row_m_new) * row_l;
        // store the new row_l and row_m
        l[lm_offset + iStart + ii] = row_l_new;
        m[lm_offset + iStart + ii] = row_m_new;
      }
    }

    __syncthreads();
  }
}

} // end namespace flash
