#pragma once
#include "common.hpp"
#include "mma.hpp"
#include <cassert>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <mma.h>

namespace flash {

/*
 * In this version, Q is distributed between the registers
 */
template <uint32_t Br, uint32_t Bc, uint32_t maxHeadDim>
__global__ void
kernel_mma_qreg_f32x4load(float const *__restrict__ Q, // query vector
                          float const *__restrict__ K, // key vector
                          float const *__restrict__ V, // value vector
                              int seq_length,              // sequence length
                              int d,                       // head_dim
                              float softmax_scale,         // 1/sqrt(d)
                              float *__restrict__ O)       // output attention
{
  constexpr uint32_t numWarps = Br / mma::Tile::M;
  assert(d % 2 == 0 && "head_dim must be even for aligned vectorized stores");
  assert(blockDim.y == numWarps);

  int batch = blockIdx.x;
  int head = blockIdx.y;
  int numHeads = gridDim.y;

  int qkv_offset = (batch * numHeads + head) * seq_length * d;

  // padded dimension d for wmma alignment
  uint32_t dp = maxHeadDim;
  assert(dp >= d);

  // shared memory for tiles
  extern __shared__ float sram[];

  // We load a Q tile into _Q, and then store Q tile in registers
  // then we load a K tile into _K (same location as Q, but maybe different size
  // Bc) and we load a V tile into _V
  float *_Q = sram;                        // size = Br * dp
  float *_K = sram;                        // size = Bc * dp
  float *_V = &sram[dp * Bc];              // size = Bc x dp
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

  const auto subtileI = warp;

  constexpr auto swizzleFuncA = common::getSkewCol<mma::Tile::M/2, mma::Tile::M/2>;
  constexpr auto swizzleFuncB = common::getSkewCol<mma::Tile::N, mma::Tile::N/2>;
  // constexpr auto swizzleFuncA = mma::returnColumn;
  // constexpr auto swizzleFuncB = mma::returnColumn;

  float mprev[mma::FragmentAccumulator::numRowsPerThread];
  float lprev[mma::FragmentAccumulator::numRowsPerThread];
  for (auto t = 0; t < mma::FragmentAccumulator::numRowsPerThread; t++) {
    mprev[t] = -INFINITY;
    lprev[t] = 0.f;
  }

  float mcur[2];
  float lcur[2];

  // Load Q tile
  auto iiStart = subtileI * mma::Tile::M;
  auto iiEnd = min((subtileI + 1) * mma::Tile::M, Brc);
  for (auto ii = iiStart; ii < iiEnd; ii++) {
    auto i = iStart + ii;
    for (uint32_t k = 4*tx; k < maxHeadDim; k += 4*warpSize) {
      auto inBounds = k < d;
      auto const *Q4 = reinterpret_cast<float4 const*>(&Q[qkv_offset + i * d + k]);
      float4 q4 = inBounds ? *Q4 : make_float4(0.f, 0.f, 0.f, 0.f);
      auto *q1 = reinterpret_cast<float*>(&q4);
      for (int r = 0; r < 4; r++) {
        auto k_sw = swizzleFuncA(ii, k + r, dp);
        _Q[ii * dp + k_sw] = q1[r];
      }
    }
  }

  // Load full Q into registers of every warp
  mma::FragmentA q_frag[numSubtilesK];
  for (uint32_t subtileK = 0; subtileK < numSubtilesK; subtileK++) {
    auto const k = subtileK * mma::Tile::K;
    mma::load_matrix_sync<swizzleFuncA>(q_frag[subtileK], _Q, iiStart, k, dp);
  }
  __syncthreads(); // this should now be here cause warps only pull rows
                   // relevant to their subtiles
                   // It would be this way if K and V did not overlap with Q

  for (int jStart = 0; jStart < seq_length; jStart += Bc) { // loop j tiles
    // Potentially cropped Bc in the last tile
    auto Bcc = min(Bc, seq_length - jStart);

    for (auto t = 0; t < mma::FragmentAccumulator::numRowsPerThread; t++) {
      mcur[t] = -INFINITY;
      lcur[t] = 0.f;
    }

    // load Kj, Vj
    for (int jj = warp; jj < Bc; jj += numWarps) {
      auto j = jStart + jj;
      // for (int k = tx; k < dp; k += blockDim.x) {
      //   auto inBounds = jj < Bcc && k < d;
      //   auto k_sw = swizzleFuncB(jj, k, dp); // swizzle column
      //   _K[jj * dp + k_sw] = inBounds ? K[qkv_offset + j * d + k] : 0.f;
      //   _V[jj * dp + k_sw] = inBounds ? V[qkv_offset + j * d + k] : 0.f;
      // }
      for (uint32_t k = 4*tx; k < maxHeadDim; k += 4*warpSize) {
        auto inBounds = k < d;
        auto const *K4 = reinterpret_cast<float4 const*>(&K[qkv_offset + j * d + k]);
        auto const *V4 = reinterpret_cast<float4 const*>(&V[qkv_offset + j * d + k]);
        float4 k4 = inBounds ? *K4 : make_float4(0.f, 0.f, 0.f, 0.f);
        float4 v4 = inBounds ? *V4 : make_float4(0.f, 0.f, 0.f, 0.f);
        auto *k1 = reinterpret_cast<float*>(&k4);
        auto *v1 = reinterpret_cast<float*>(&v4);
        for (int r = 0; r < 4; r++) {
          auto k_sw = swizzleFuncB(jj, k + r, dp);
          _K[jj * dp + k_sw] = k1[r];
          _V[jj * dp + k_sw] = v1[r];
        }
      }
    }
    __syncthreads();

    // This allows to remove __syncthreads at the end of the loop,
    // but it actually makes the kernel slower :-(

    // for (int jj = warp; jj < Bc; jj += numWarps) {
    //   auto j = jStart + jj;
    //   for (int k = tx; k < dp; k += blockDim.x) {
    //     auto inBounds = jj < Bcc && k < d;
    //     auto k_sw = swizzleFuncB(jj, k, dp); // swizzle column
    //     _K[jj * dp + k_sw] = inBounds ? K[qkv_offset + j * d + k] : 0.f;
    //   }
    // }
    // __syncthreads();
    // for (int jj = warp; jj < Bc; jj += numWarps) {
    //   auto j = jStart + jj;
    //   for (int k = tx; k < dp; k += blockDim.x) {
    //     auto inBounds = jj < Bcc && k < d;
    //     auto k_sw = swizzleFuncB(jj, k, dp); // swizzle column
    //     _V[jj * dp + k_sw] = inBounds ? V[qkv_offset + j * d + k] : 0.f;
    //   }
    // }

    // Compute Sij = Qij * Kj
    mma::FragmentB<mma::Layout::col_major> k_frag;
    mma::FragmentAccumulator s_frag;

    // Each warp will compute a portion of j subtiles
    // for now, I assume only one warp does it though since no reduction
    for (auto subtileJ = 0; subtileJ < numSubtilesJ; subtileJ++) {
      auto const ii = subtileI * mma::Tile::M;
      auto const jj = subtileJ * mma::Tile::N;

      mma::fill_fragment(s_frag, 0.f);
      for (auto subtileK = 0; subtileK < numSubtilesK; subtileK++) {
        auto k = subtileK * mma::Tile::K;
        mma::load_matrix_sync<swizzleFuncB>(k_frag, _K, jj, k, dp);
        mma::mma_sync(s_frag, q_frag[subtileK], k_frag, s_frag);
      }
      for (int t = 0; t < s_frag.registersPerThread; t++) {
        s_frag.reg[t] *= softmax_scale;
      }

      // Each accumulator really only stores values in 2 rows per thread
      // we can store them locally instead of doing a separate reduction
      // reduce row_max
      mma::threadReduceByRow<common::float_max>(s_frag, mcur, Bcc - jj);
      mma::store_matrix_sync<swizzleFuncA>(_S, ii, jj, Bc, s_frag);
    }
    mma::warpReduceFragAccumulatorRowValue<common::float_max>(mcur);

    // Compute row sum
    // Here I assume again one warp X
    // But we now can reuse K for shared memory sync
    for (uint32_t frag_jj = 0; frag_jj < Bc; frag_jj += mma::Tile::K) {
      auto const frag_ii = subtileI * mma::Tile::M;
      mma::load_matrix_sync<swizzleFuncA>(s_frag, _S, frag_ii, frag_jj, Bc);
      for (int r = 0; r < s_frag.registersPerThread; r++) {
        auto ii = frag_ii + s_frag.threadRow(r);
        auto jj = frag_jj + s_frag.threadCol(r);
        auto const row_m = mcur[s_frag.rowIndex(r)];
        auto Pij = __expf(s_frag.reg[r] - row_m);
        Pij = (frag_ii < Brc && jj < Bcc) ? Pij : 0.f;
        s_frag.reg[r] = Pij;
      }
      mma::threadReduceByRow<common::float_add>(s_frag, lcur, Bcc - frag_jj);
      mma::store_matrix_sync<swizzleFuncA>(_S, frag_ii, frag_jj, Bc, s_frag);
    }
    // Compute row_sum: warp-reduce lcur by row
    mma::warpReduceFragAccumulatorRowValue<common::float_add>(lcur);

    mma::FragmentA p_frag;
    mma::FragmentB<mma::Layout::row_major> v_frag;
    mma::FragmentAccumulator pv_frag;

    for (auto subtileK = 0; subtileK < numSubtilesK; subtileK++) {
      auto const frag_ii = subtileI * mma::Tile::M;
      auto const frag_k = subtileK * mma::Tile::N;
      fill_fragment(pv_frag, 0.f);

      for (uint32_t frag_jj = 0; frag_jj < Bc; frag_jj += mma::Tile::K) {
        mma::load_matrix_sync<swizzleFuncA>(p_frag, _S, frag_ii, frag_jj, Bc);
        mma::load_matrix_sync<swizzleFuncB>(v_frag, _V, frag_jj, frag_k, dp);
        mma::mma_sync(pv_frag, p_frag, v_frag, pv_frag);
      }
      // now we have pv_frag and we can directly write output
      // This should be optimal because a row of fragment C is:
      // 0 0 1 1 2 2 3 3 (8x4 = 32 bytes == size of a sector)
      float rO[pv_frag.registersPerThread];
      for (auto r = 0; r < pv_frag.registersPerThread; r += 2) {
        auto fragRow = mma::FragmentAccumulator::threadRow(r);
        auto fragCol = mma::FragmentAccumulator::threadCol(r);
        auto ii = frag_ii + fragRow;
        auto i = iStart + ii;
        auto k = frag_k + fragCol;
        if (ii < Brc && k < d) {
          auto const *src = reinterpret_cast<float2 const*>(&O[qkv_offset + i * d + k]);
          auto *dst = reinterpret_cast<float2*>(&rO[r]);
          *dst = *src;
        }
      }
      for (auto r = 0; r < pv_frag.registersPerThread; r++) {
        auto PVik = pv_frag.reg[r];
        auto thread_lm_row = pv_frag.rowIndex(r);
        auto const row_m = mcur[thread_lm_row];
        auto const row_l = lcur[thread_lm_row];
        auto const row_m_prev = mprev[thread_lm_row];
        auto const row_l_prev = lprev[thread_lm_row];
        auto const row_m_new = common::float_max(row_m_prev, row_m);
        auto const row_l_new = __expf(row_m_prev - row_m_new) * row_l_prev +
                               __expf(row_m - row_m_new) * row_l;
        rO[r] = ((row_l_prev * __expf(row_m_prev - row_m_new) * rO[r]) +
              (__expf(row_m - row_m_new) * PVik)) /
              row_l_new;
      }
      for (auto r = 0; r < pv_frag.registersPerThread; r += 2) {
        auto fragRow = mma::FragmentAccumulator::threadRow(r);
        auto fragCol = mma::FragmentAccumulator::threadCol(r);
        auto ii = frag_ii + fragRow;
        auto i = iStart + ii;
        auto k = frag_k + fragCol;
        if (ii < Brc) {
          if (k + 1 < d) {
            auto *dst = reinterpret_cast<float2*>(&O[qkv_offset + i * d + k]);
            auto const *src = reinterpret_cast<float2*>(&rO[r]);
            *dst = *src;
          }
          else if (k < d) {
           O[qkv_offset + i * d + k] = rO[r];
          }
        }
      }
    }
    // store new l and m!!!!
    for (auto r = 0; r < mma::FragmentAccumulator::numRowsPerThread; r++) {
      auto const row_m = mcur[r];
      auto const row_l = lcur[r];
      auto const row_m_prev = mprev[r];
      auto const row_l_prev = lprev[r];
      float row_m_new = common::float_max(row_m_prev, row_m);
      float row_l_new = __expf(row_m_prev - row_m_new) * row_l_prev +
                        __expf(row_m - row_m_new) * row_l;
      mprev[r] = row_m_new;
      lprev[r] = row_l_new;
    }

    __syncthreads();
  }
}

} // end namespace flash
