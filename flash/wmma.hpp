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

} // namespace flash::wmma
