#include <stdint.h>
#include <stdio.h>
#include <type_traits>

namespace flash::mma {

struct Tile {
  static constexpr uint M = 16;
  static constexpr uint N = 8;
  static constexpr uint K = 8;
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

__device__ inline void load_matrix_sync(FragmentA &frag, float const *ptr,
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
  auto lane = threadIdx.x;
  auto group = lane >> 2;
  auto member = lane % 4;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    uint32_t row = (i == 0 || i == 2) ? group : group + 8;
    uint32_t col = (i == 0 || i == 1) ? member : member + 4;
    asm("cvt.rna.tf32.f32  %0, %1;\n"
        : "=r"(frag.reg[i])
        : "f"(ptr[row * stride + col]));
  }
}

template <Layout layout>
__device__ inline void load_matrix_sync(FragmentB<layout> &frag, float *ptr,
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
    uint32_t row = (i == 0) ? member : member + 4;
    uint32_t col = group;
    if constexpr (layout == Layout::row_major) {
      asm("cvt.rna.tf32.f32  %0, %1;\n"
          : "=r"(frag.reg[i])
          : "f"(ptr[row * stride + col]));
    } else {
      asm("cvt.rna.tf32.f32  %0, %1;\n"
          : "=r"(frag.reg[i])
          : "f"(ptr[col * stride + row]));
    }
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

// template <typename T>
// concept FragmentType = requires {
//   requires std::is_same_v<decltype(T::size), const uint8_t>;
//   requires T::size > 0;
//   requires std::is_same_v<decltype(T{}.reg), uint32_t[T::size]>;
// };

// template <FragmentType frag_t>
__device__ inline void fill_fragment(FragmentAccumulator &frag, float value) {
  for (auto i = 0; i < frag.size; i++)
    frag.reg[i] = value;
}

__device__ inline void store_matrix_sync(float *ptr, FragmentAccumulator &frag,
                                         uint32_t stride) {
  /*
  groupID           = %laneid >> 2
  threadID_in_group = %laneid % 4
  row =      groupID                            for c0 and c1
          groupID + 8                          for c2 and c3
  col =  (threadID_in_group * 2) + (i & 0x1)    for ci   where i = {0,..,3}
  */
  auto lane = threadIdx.x;
  auto group = lane >> 2;
  auto member = lane % 4;

#pragma unroll
  for (int i = 0; i < 4; i++) {
    uint32_t row = (i < 2) ? group : group + 8;
    uint32_t col = 2 * member + (i & 0x1);
    ptr[stride * row + col] = frag.reg[i];
  }
}

} // namespace flash::mma
