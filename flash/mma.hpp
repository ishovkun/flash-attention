#pragma once
#include <stdint.h>
#include <stdio.h>
#include <type_traits>
#include "common.hpp"

namespace flash::mma {

struct Tile {
  static constexpr uint32_t M = 16;
  static constexpr uint32_t N = 8;
  static constexpr uint32_t K = 8;
};

struct FragmentA {
  static constexpr uint8_t registersPerThread = 4;
  uint32_t reg[registersPerThread];

  static __device__ uint32_t threadRow(uint32_t register_idx) {
    auto lane = threadIdx.x;
    auto group = lane >> 2;
    return (register_idx % 2 == 0) ? group : group + 8;
  }
  static __device__ uint32_t threadCol(uint32_t register_idx) {
    auto lane = threadIdx.x;
    // auto group = lane >> 2;
    auto member = lane % 4;
    return (register_idx < 2) ? member : member + 4;
  }
};

enum class Layout {
  row_major,
  col_major,
};

template <Layout layout> struct FragmentB {
  static constexpr uint8_t registersPerThread = 2;
  uint32_t reg[registersPerThread];

  /*
  groupID           = %laneid >> 2
  threadID_in_group = %laneid % 4
  row =    threadID_in_group         for b0
           threadID_in_group + 4     for b1
  col =  groupID
  */

  static __device__ uint32_t threadRow(uint32_t register_idx) {
    auto lane = threadIdx.x;
    auto member = lane % 4;
    return (register_idx == 0) ? member : member + 4;
  }

  static __device__ uint32_t threadCol(uint32_t register_idx) {
    auto lane = threadIdx.x;
    auto group = lane >> 2;
    return group;
  }
};


/*
 groupID           = %laneid >> 2
 threadID_in_group = %laneid % 4
 row =      groupID      for c0 and c1
            groupID + 8  for c2 and c3
 col =  (threadID_in_group * 2) + (i & 0x1)    for ci   where i = {0,..,3}
 Example: float
   | 0  1  2  3  4  5  6  7
 --------------------------
 0 | 0  0  1  1  2  2  3  3
 1 | 4  4  5  5  6  6  7  7
 2 | 8  8  9  9  10 10 11 11
 3 | 2 12 13 13 14 14 15 15
 4 | 6 16 17 17 18 18 19 19
 5 | 0 20 21 21 22 22 23 23
 6 | 4 24 25 25 26 26 27 27
 7 | 8 28 29 29 30 30 31 31
 8 |   0  1  1  2  2  3  3
 9 |   4  5  5  6  6  7  7
 10|   8  9  9  10 10 11 11
 11| 2 12 13 13 14 14 15 15
 12| 6 16 17 17 18 18 19 19
 13| 0 20 21 21 22 22 23 23
 14| 4 24 25 25 26 26 27 27
 15| 8 28 29 29 30 30 31 31
*/
struct FragmentAccumulator {
  static constexpr uint8_t registersPerThread = 4;
  static constexpr uint8_t numRowsPerThread = 2;
  float reg[registersPerThread];

  static __device__ uint32_t threadRow(uint32_t register_idx) {
    auto lane = threadIdx.x;
    auto group = lane >> 2;
    return (register_idx < 2) ? group : group + 8;
  }

  static __device__ uint32_t rowIndex(uint32_t register_idx) {
    return (register_idx < 2) ? 0 : 1;
  }

  static __device__ uint32_t threadCol(uint32_t register_idx) {
    auto lane = threadIdx.x;
    auto member = lane % 4;
    return 2 * member + (register_idx & 0x1);
  }
};

template <typename F>
concept U32ColumnIndexFunc =
    requires(F f, uint32_t row, uint32_t col, uint32_t stride) {
      { f(row, col, stride) } -> std::same_as<uint32_t>;
    };

inline __device__ uint32_t returnColumn(uint32_t row, uint32_t col,
                                        uint32_t stride) {
  return col;
}

template <U32ColumnIndexFunc auto returnColumnFunc = returnColumn>
inline __device__ auto load_matrix_sync(FragmentA &frag, float const *ptr,
                                        uint32_t tileFirstRow, uint32_t tileFirstCol,
                                        uint32_t stride) {
#pragma unroll
  for (uint32_t i = 0; i < frag.registersPerThread; i++) {
    uint32_t row = tileFirstRow + frag.threadRow(i);
    uint32_t col = tileFirstCol + frag.threadCol(i);
    // potentially swizzle if a swizzle function is passed
    col = returnColumnFunc(row, col, stride);
    // convert f32 -> tf32 & load into registers
    asm("cvt.rna.tf32.f32  %0, %1;\n"
        : "=r"(frag.reg[i])
        : "f"(ptr[row * stride + col]));
  }
}

template<U32ColumnIndexFunc auto returnColumnFunc = returnColumn, Layout layout>
__device__ inline void load_matrix_sync(FragmentB<layout> &frag,
                                        float const *ptr,
                                        uint32_t tileFirstRow,
                                        uint32_t tileFirstCol,
                                        uint32_t stride) {
#pragma unroll
  for (int i = 0; i < frag.registersPerThread; i++) {
    auto tileRow = frag.threadRow(i);
    auto tileCol = frag.threadCol(i);
    if constexpr (layout == Layout::col_major) {
      common::swap(tileRow, tileCol);
    }
    uint32_t row = tileFirstRow + tileRow;
    uint32_t col = tileFirstCol + tileCol;
    col = returnColumnFunc(row, col, stride);
    asm("cvt.rna.tf32.f32  %0, %1;\n"
        : "=r"(frag.reg[i])
        : "f"(ptr[row * stride + col]));
  }
}

template <U32ColumnIndexFunc auto returnColumnFunc = returnColumn>
inline __device__ auto load_matrix_sync(FragmentAccumulator &frag, float const *ptr,
                                        uint32_t fragFirstRow, uint32_t fragFirstCol,
                                        uint32_t stride) {
  for (int i = 0; i < FragmentAccumulator::registersPerThread; i++) {
    auto fragRow = frag.threadRow(i);
    auto fragCol = frag.threadCol(i);
    uint32_t row = fragFirstRow + fragRow;
    uint32_t col = fragFirstCol + fragCol;
    col = returnColumnFunc(row, col, stride);
    frag.reg[i] = ptr[stride * row + col];
  }
}

template <Layout layout>
__device__ inline void mma_sync(FragmentAccumulator &D, FragmentA const &A,
                                FragmentB<layout> const &B,
                                FragmentAccumulator const &C) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
               : "=f"(D.reg[0]), "=f"(D.reg[1]), "=f"(D.reg[2]), "=f"(D.reg[3])
               : "r"(A.reg[0]), "r"(A.reg[1]), "r"(A.reg[2]), "r"(A.reg[3]),
                 "r"(B.reg[0]), "r"(B.reg[1]), "f"(C.reg[0]), "f"(C.reg[1]),
                 "f"(C.reg[2]), "f"(C.reg[3]));
}

__device__ inline void fill_fragment(FragmentAccumulator &frag, float value) {
  for (auto i = 0; i < frag.registersPerThread; i++)
    frag.reg[i] = value;
}

template<U32ColumnIndexFunc auto returnColumnFunc = returnColumn>
__device__ inline void store_matrix_sync(float *ptr, uint32_t fragFirstRow,
                                          uint32_t fragFirstCol, uint32_t stride,
                                          FragmentAccumulator &frag) {
#pragma unroll
  for (int i = 0; i < frag.registersPerThread; i++) {
    auto fragRow = frag.threadRow(i);
    auto fragCol = frag.threadCol(i);
    uint32_t row = fragFirstRow + fragRow;
    uint32_t col = fragFirstCol + fragCol;
    col = returnColumnFunc(row, col, stride);
    ptr[stride * row + col] = frag.reg[i];
  }
}

template <common::F32BinaryFunc auto binaryFunc>
__device__ void threadReduceByRow(FragmentAccumulator const &frag, float (&ret)[2], uint32_t maxCol = Tile::N) {
  for (uint32_t i = 0; i < FragmentAccumulator::registersPerThread; i++) {
    auto fragCol = frag.threadCol(i);
    uint32_t irow = frag.rowIndex(i);
    auto value = (fragCol < maxCol) ? frag.reg[i] : common::F32BinaryFuncTraits<binaryFunc>::sentinel;
    ret[irow] = binaryFunc(value, ret[irow]);
  }
}

template <common::F32BinaryFunc auto binaryFunc>
__device__ void warpReduceFragAccumulatorRowValue(float (&ret)[2]) {
  auto lane = threadIdx.x;
  auto member = lane % 4;
  for (int s = 2; s > 0; s /= 2) {
    auto tmp1 = __shfl_down_sync(UINT32_MAX, ret[0], s);
    auto tmp2 = __shfl_down_sync(UINT32_MAX, ret[1], s);
    if (member < s) {
      ret[0] = binaryFunc(ret[0], tmp1);
      ret[1] = binaryFunc(ret[1], tmp2);
    }
  }
  ret[0] = __shfl_sync(UINT32_MAX, ret[0], lane - member);
  ret[1] = __shfl_sync(UINT32_MAX, ret[1], lane - member);
}


} // namespace flash::mma
