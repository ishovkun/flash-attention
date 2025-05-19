#pragma once
#include "common.hpp"
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
  static constexpr uint8_t size = 4;
  uint32_t reg[size];
};

enum class Layout {
  row_major,
  col_major,
};

template <Layout layout> struct FragmentB {
  static constexpr uint8_t size = 2;
  uint32_t reg[size];
};

struct FragmentAccumulator {
  static constexpr uint8_t size = 4;
  float reg[size];
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
  auto lane = threadIdx.x;
  auto group = lane >> 2;
  auto member = lane % 4;
#pragma unroll
  for (uint32_t i = 0; i < 4; i++) {
    uint32_t tileRow = (i == 0 || i == 2) ? group : group + 8;
    uint32_t tileCol = (i == 0 || i == 1) ? member : member + 4;
    uint32_t row = tileFirstRow + tileRow;
    uint32_t col = tileFirstCol + tileCol;
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
  /*
  groupID           = %laneid >> 2
  threadID_in_group = %laneid % 4
  row =    threadID_in_group         for b0
           threadID_in_group + 4     for b1
  col =  groupID
  */
  auto lane = threadIdx.x;
  auto group = lane >> 2;
  auto member = lane % 4;
#pragma unroll
  for (int i = 0; i < 2; i++) {
    uint32_t tileRow = (i == 0) ? member : member + 4;
    uint32_t tileCol = group;
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
  auto lane = threadIdx.x;
  auto group = lane >> 2;
  auto member = lane % 4;
  for (int i = 0; i < FragmentAccumulator::size; i++) {
    uint32_t fragRow = (i < 2) ? group : group + 8;
    uint32_t fragCol = 2 * member + (i & 0x1);
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
  for (auto i = 0; i < frag.size; i++)
    frag.reg[i] = value;
}

template<U32ColumnIndexFunc auto returnColumnFunc = returnColumn>
__device__ inline void store_matrix_sync(float *ptr, uint32_t fragFirstRow,
                                          uint32_t fragFirstCol, uint32_t stride,
                                          FragmentAccumulator &frag) {
  /*
   groupID           = %laneid >> 2
   threadID_in_group = %laneid % 4
   row =      groupID      for c0 and c1
              groupID + 8  for c2 and c3
   col =  (threadID_in_group * 2) + (i & 0x1)    for ci   where i = {0,..,3}
  */
  auto lane = threadIdx.x;
  auto group = lane >> 2;
  auto member = lane % 4;

  #pragma unroll
  for (int i = 0; i < 4; i++) {
    uint32_t fragRow = (i < 2) ? group : group + 8;
    uint32_t fragCol = 2 * member + (i & 0x1);
    uint32_t row = fragFirstRow + fragRow;
    uint32_t col = fragFirstCol + fragCol;
    col = returnColumnFunc(row, col, stride);
    ptr[stride * row + col] = frag.reg[i];
  }
}

template <common::F32BinaryFunc auto binaryFunc>
__device__ void threadReduceByRow(FragmentAccumulator const &frag, float (&ret)[2], uint32_t maxCol = Tile::N) {
  auto lane = threadIdx.x;
  // auto group = lane >> 2;
  auto member = lane % 4;

  for (uint32_t i = 0; i < FragmentAccumulator::size; i++) {
    uint32_t fragCol = 2 * member + (i & 0x1);
    uint32_t irow = (i < 2) ? 0 : 1;
    auto value = (fragCol < maxCol) ? frag.reg[i] : common::F32BinaryFuncTraits<binaryFunc>::default_value;
    ret[irow] = binaryFunc(value, ret[irow]);
  }
}

// template <common::F32BinaryFunc auto binaryFunc>
// __device__ void warpReduceByRow(FragmentAccumulator const &frag, float (&ret)[2], uint32_t maxCol = Tile::N) {
//   auto lane = threadIdx.x;
//   // auto group = lane >> 2;
//   auto member = lane % 4;

//   threadReduceByRow<binaryFunc>(frag, ret, maxCol);

//   for (int s = 2; s > 0; s /= 2) {
//     // TODO: send float2 instead of two instructions
//     auto tmp1 = __shfl_down_sync(UINT32_MAX, ret[0], s);
//     auto tmp2 = __shfl_down_sync(UINT32_MAX, ret[1], s);
//     if (member < s) {
//       ret[0] = binaryFunc(ret[0], tmp1);
//       ret[1] = binaryFunc(ret[1], tmp2);
//     }
//   }
//   // distribute to all threads
//   // TODO: send float
//   for (int i = 0; i < 2; i++) {
//     // check this when I'm sober
//     ret[i] = __shfl_sync(UINT32_MAX, ret[i], lane - lane % 4);
//   }
// }

template <common::F32BinaryFunc auto binaryFunc>
__device__ void warpSuperReduceByRow(float (&ret)[2]) {
  auto lane = threadIdx.x;
  for (int s = 2; s > 0; s /= 2) {
    auto member = lane % 4;
    auto tmp1 = __shfl_down_sync(UINT32_MAX, ret[0], s);
    auto tmp2 = __shfl_down_sync(UINT32_MAX, ret[1], s);
    if (member < s) {
      ret[0] = binaryFunc(ret[0], tmp1);
      ret[1] = binaryFunc(ret[1], tmp2);
    }
  }
  ret[0] = __shfl_sync(UINT32_MAX, ret[0], lane - lane % 4);
  ret[1] = __shfl_sync(UINT32_MAX, ret[1], lane - lane % 4);
}


} // namespace flash::mma
