#pragma once
#include <cuda.h>

namespace flash::common {

constexpr int warpSize = 32;
constexpr int maxBlockSize = 1024;

inline float __device__ float_max(float a, float b) { return a > b ? a : b; }

inline float __device__ float_add(float a, float b) { return a + b; }

template <auto binaryFunc = float_add>
__device__ float warpReduce(float value) {
  constexpr int warpSize = 32;
  int lane = threadIdx.x % warpSize;
  for (int s = warpSize / 2; s > 0; s >>= 1) {
    auto tmp = __shfl_down_sync(UINT32_MAX, value, s);
    if (lane < s)
      value = binaryFunc(value, tmp);
  }
  // broadcast value from lane 0
  return __shfl_sync(UINT32_MAX, value, 0);
}

inline constexpr int __host__ __device__ ceil_div(int a, int b) {
  return (a + b - 1) / b;
}

inline int __host__ __device__ nextMultiple(int a, int b) {
  return ceil_div(a, b) * b;
}

inline int __host__ __device__ prevMultiple(int a, int b) {
  return a - a % b;
}

} // end namespace flash::common
