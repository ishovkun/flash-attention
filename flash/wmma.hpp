#pragma once
#include <mma.h>

namespace flash::wmma {

using nvcuda::wmma::__float_to_tf32;
using nvcuda::wmma::load_matrix_sync;
using nvcuda::wmma::mem_row_major;
using nvcuda::wmma::mma_sync;
using nvcuda::wmma::store_matrix_sync;
using nvcuda::wmma::fill_fragment;

// A: M x K
// B: K x N
// C: M x N
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 8;

// using namespace nvcuda;
using fragA_t = nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N,
                                       WMMA_K, nvcuda::wmma::precision::tf32,
                                       nvcuda::wmma::row_major>;
// note col_major for transpose
using fragB_cm_t =
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                           nvcuda::wmma::precision::tf32,
                           nvcuda::wmma::col_major>;
using fragB_rm_t =
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                           nvcuda::wmma::precision::tf32,
                           nvcuda::wmma::row_major>;
using fragC_t = nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M,
                                       WMMA_N, WMMA_K, float>;

struct FragmentA {
  uint32_t reg[4];
};

enum class Layout {
  row_major,
  col_major,
};
template <Layout layout> struct FragmentB {
  uint32_t reg[4];
};

struct FragmentAccumulator {
  uint32_t reg[4];
};

__device__ inline void load_matrix_sync(FragmentA &frag, float *ptr,
                                        uint32_t stride) {
  /*
  A fragment:
  groupID           = %laneid >> 2
  threadID_in_group = %laneid % 4
  row =      groupID            for a0 and a2
             groupID + 8        for a1 and a3
  col =  threadID_in_group       for a0 and a1
         threadID_in_group + 4   for a2 and a3
  */
  auto laneid = threadIdx.x;
  auto groupID = laneid >> 2;
  auto threadID_in_group = laneid % 4;
  frag.reg[0] = ptr[groupID * stride + threadID_in_group];
  frag.reg[1] = ptr[(groupID + 8) * stride + threadID_in_group];
  frag.reg[2] = ptr[groupID * stride + threadID_in_group + 4];
  frag.reg[3] = ptr[(groupID + 8) * stride + threadID_in_group + 4];
  /*
  wmma.mma.sync.aligned.alayout.blayout.shape.f32.atype.btype.f32 d, a, b, c;
    .alayout = {.row, .col};
    .blayout = {.row, .col};
    .shape   = {.m16n16k8 };
    .atype   = {.tf32 };
    .btype   = {.tf32};
    wmma.mma.sync.aligned.row.col.m16n16k8.f32.tr32.tf32.f32 d, a, b, c;
  */
}

template <Layout layout>
__device__ inline void load_matrix_sync(FragmentB<layout> &frag, float *ptr,
                                        uint32_t stride) {
  /*
  groupID           = %laneid >> 2
  threadID_in_group = %laneid % 4
  row =    threadID_in_group         for b0
         threadID_in_group + 4       for b1
  col =  groupID
  */
  auto laneid = threadIdx.x;
  auto groupID = laneid >> 2;
  auto threadID_in_group = laneid % 4;
  if constexpr (layout == Layout::col_major) {
    frag.reg[0] = ptr[groupID * stride + threadID_in_group];
    frag.reg[1] = ptr[(groupID + 8) * stride + threadID_in_group];
    frag.reg[2] = ptr[groupID * stride + threadID_in_group + 4];
    frag.reg[3] = ptr[(groupID + 8) * stride + threadID_in_group + 4];
  }
}

__device__ inline void load_matrix_sync(FragmentAccumulator &frag, float *ptr,
                                        uint32_t stride) {
  /*
  groupID           = %laneid >> 2
  threadID_in_group = %laneid % 4
  row =      groupID                            for c0 and c1
          groupID + 8                          for c2 and c3
  col =  (threadID_in_group * 2) + (i & 0x1)    for ci   where i = {0,..,3}
  */
}

template <Layout layout>
__device__ inline void mma_sync(FragmentAccumulator &D, FragmentA const &A,
                                FragmentB<layout> const &B,
                                FragmentAccumulator const &C) {
  // Compute alternate floating point precision wmma
  // .reg .b32 a<2>, b<2>, c<8>, d<8>;
  // wmma.mma.sync.aligned.m16n16k8.row.col.f32.tf32.tf32.f32
  //         {d0, d1, d2, d3, d4, d5, d6, d7},
  //         {a0, a1, a2, a3}, {b0, b1, b2, b3},
  //         {c0, c1, c2, c3, c4, c5, c6, c7};
}

} // namespace flash::wmma
