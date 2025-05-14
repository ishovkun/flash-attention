#pragma once
#include "common.hpp"
#include "mma.hpp"
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <mma.h>

namespace flash {

template <uint32_t Br, uint32_t Bc, uint32_t numWarps>
__global__ void kernel_mma_sync_swizzle(
    float const *__restrict__ Q, // query vector
    float const *__restrict__ K, // key vector
    float const *__restrict__ V, // value vector
    int seq_length,              // sequence length
    int d,                       // head_dim
    float softmax_scale,         // 1/sqrt(d)
    float *__restrict__ l,       // storage temp for row \sum exp(S)
    float *__restrict__ m,       // storage temp for row \max S
    float *__restrict__ O)       // output attention
{
  int batch = blockIdx.x;
  int head = blockIdx.y;
  int numHeads = gridDim.y;

  int qkv_offset = (batch * numHeads + head) * seq_length * d;
  int lm_offset = (batch * numHeads + head) * seq_length;

  // padded dimension d for wmma alignment
  uint32_t dp = common::nextMultiple(d, mma::Tile::N);

  // bug is here
  constexpr auto rowsPerWarp = common::ceil_div(Br, numWarps);
  float mcur[rowsPerWarp];

  // bug: Br != Bc, but we save PV into K (Bc x d)

  // shared memory for tiles
  extern __shared__ float sram[];
  float *_Q = sram;                      // size = Br x d
  float *_K = &sram[Br * dp];            // size = Bc x d
  float *_V = &sram[dp * (Bc + Br)];     // size = Bc x d
  float *_S = &sram[dp * (Br + 2 * Bc)]; // size = Br x Bc

  auto const tx = threadIdx.x;
  auto const ty = threadIdx.y;
  auto const warp = ty;

  auto const iStart = blockIdx.z * Br;
  auto const iEnd = min(iStart + Br, seq_length);
  auto const Brc = min(Br, seq_length - iStart);

  auto const iiStart = warp * rowsPerWarp;
  auto const iiEnd = min((warp + 1) * rowsPerWarp, Brc);

  constexpr auto swizzleFuncA = common::getSkewCol<mma::Tile::M/2, mma::Tile::M/2>;
  constexpr auto swizzleFuncB = common::getSkewCol<mma::Tile::N, mma::Tile::N/2>;

  // set l and m to default values
  for (int i = iStart + warp * warpSize + tx; i < iEnd; i += warpSize * numWarps) {
    l[lm_offset + i] = 0.f;
    m[lm_offset + i] = -INFINITY;
  }

  // Load Q tile
  for (int ii = warp; ii < Brc; ii += numWarps) {
    auto i = iStart + ii;
    for (int k = tx; k < dp; k += blockDim.x) {
      auto inBounds = k < d;
      auto k_sw = swizzleFuncA(ii, k, dp);
      _Q[ii*dp + k_sw] = inBounds ? Q[qkv_offset + i * d + k] : 0.f;
    }
  }

  for (int jStart = 0; jStart < seq_length; jStart += Bc) { // loop j tiles
    // Potentially cropped Bc in the last tile
    auto Bcc = min(Bc, seq_length - jStart);

    // load Kj, Vj
    for (int jj = warp; jj < Bc; jj += numWarps) {
      auto j = jStart + jj;
      for (int k = tx; k < dp; k += warpSize) {
        auto inBounds = jj < Bcc && k < d;
        // auto k_sw = common::getSkewCol<mma::Tile::N>(jj, k, dp); // swizzle column
        auto k_sw = swizzleFuncB(jj, k, dp); // swizzle column
        _K[jj*dp + k_sw] = inBounds ? K[qkv_offset + j * d + k] : 0.f;
        _V[jj*dp + k_sw] = inBounds ? V[qkv_offset + j * d + k] : 0.f;
      }
    }

    __syncthreads();

    // Compute Sij
    mma::FragmentA q_frag;
    mma::FragmentB<mma::Layout::col_major> k_frag;
    mma::FragmentAccumulator s_frag;

    constexpr auto numSubtilesI = common::ceil_div(Br, mma::Tile::M);
    constexpr auto numSubtilesJ = common::ceil_div(Bc, mma::Tile::N);
    auto numSubtilesQK = numSubtilesI * numSubtilesJ;
    for (uint32_t subTile = warp; subTile < numSubtilesQK; subTile += numWarps) {
      mma::fill_fragment(s_frag, 0.f);
      auto subtileI = subTile / numSubtilesJ;
      auto subtileJ = subTile % numSubtilesJ;
      auto ii = subtileI * mma::Tile::M;
      auto jj = subtileJ * mma::Tile::N;
      for (int k = 0; k < d; k += mma::Tile::K) {
        mma::load_matrix_sync<swizzleFuncA>(q_frag, _Q, ii, k, dp);
        mma::load_matrix_sync<swizzleFuncB>(k_frag, _K, jj, k, dp);
        mma::mma_sync(s_frag, q_frag, k_frag, s_frag);
      }
      // apply scaling
      for (int t = 0; t < s_frag.size; t++)
        s_frag.reg[t] *= softmax_scale;
      mma::store_matrix_sync<swizzleFuncA>(_S, ii, jj, Bc, s_frag);
    }
    __syncthreads();

    // P = exp(S)
    for (auto ii = iiStart; ii < iiEnd; ii++) {
      float row_m = -INFINITY;
      for (int jj = tx; jj < Bcc; jj += blockDim.x) {
        auto jj_sw = swizzleFuncA(ii, jj, Bc);
        row_m = common::float_max(row_m, _S[Bc * ii + jj_sw]);
      }
      row_m = common::warpReduce<common::float_max>(row_m);
      mcur[ii - iiStart] = row_m;
      for (int jj = tx; jj < Bc; jj += blockDim.x) {
        auto jj_sw = swizzleFuncA(ii, jj, Bc);
        float Pij = __expf(_S[Bc * ii + jj_sw] - row_m);
        _S[Bc * ii + jj_sw] = (ii < Brc && jj < Bcc) ? Pij : 0.f;
      }
    }
    __syncthreads();

    float *_PV = _K; // reuse _K to store PV

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
      for (uint32_t jj = 0; jj < Bc; jj += mma::Tile::K) {
        mma::load_matrix_sync<swizzleFuncA>(p_frag, _S, ii, jj, Bc);
        mma::load_matrix_sync<swizzleFuncB>(v_frag, _V, jj, k, dp);
        mma::mma_sync(pv_frag, p_frag, v_frag, pv_frag);
      }
      mma::store_matrix_sync<swizzleFuncA>(_PV, ii, k, dp, pv_frag);
    }
    __syncthreads();

    // Write O,l, and m to global memory
    for (auto ii = iiStart; ii < iiEnd; ii++) {
      auto i = iStart + ii;
      auto row_m = mcur[ii - iiStart];

      // compute row_l
      float row_l = 0.f;
      for (int jj = tx; jj < Bcc; jj += warpSize) {
        auto jj_sw = swizzleFuncA(ii, jj, Bc);
        row_l += _S[Bc * ii + jj_sw];
      }
      row_l = common::warpReduce<common::float_add>(row_l);

      float row_m_prev = m[lm_offset + i];
      float row_l_prev = l[lm_offset + i];
      float row_m_new = common::float_max(row_m_prev, row_m);
      float row_l_new = __expf(row_m_prev - row_m_new) * row_l_prev +
                        __expf(row_m - row_m_new) * row_l;

      // compute O
      for (int k = tx; k < d; k += blockDim.x) {
        auto k_sw = swizzleFuncA(ii, k, dp);
        auto pv = _PV[ii * dp + k_sw];
        O[qkv_offset + i * d + k] =
            ((row_l_prev * __expf(row_m_prev - row_m_new) *
              O[qkv_offset + i * d + k]) +
             (__expf(row_m - row_m_new) * pv)) /
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
