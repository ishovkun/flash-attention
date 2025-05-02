#pragma once
#include <mma.h>

namespace flash::constants {

constexpr int warpSize = 32;
constexpr int maxBlockSize = 1024;
// A: M x K
// B: K x N
// C: M x N
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 8;

using namespace nvcuda;
using fragA_t = wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                               wmma::precision::tf32, wmma::row_major>;
// note col_major for transpose
using fragB_cm_t = wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                                  wmma::precision::tf32, wmma::col_major>;
using fragB_rm_t = wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                                  wmma::precision::tf32, wmma::row_major>;
using fragC_t = wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>;

} // namespace flash::constants
