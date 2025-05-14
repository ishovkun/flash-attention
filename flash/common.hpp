#pragma once
#include <cuda.h>

namespace flash::common {

constexpr int warpSize = 32;
constexpr int maxBlockSize = 1024;

template <typename F>
concept F32BinaryFunc = requires(F f, float a, float b) {
  { f(a, b) } -> std::same_as<float>;
};

inline float __device__ float_max(float a, float b) { return a > b ? a : b; }

inline float __device__ float_add(float a, float b) { return a + b; }

template <F32BinaryFunc auto binaryFunc = float_add>
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

inline int __host__ __device__ prevMultiple(int a, int b) { return a - a % b; }

inline __device__ bool isPowerOf2(uint32_t x) { return (x & (x - 1)) == 0; }

inline __device__ uint32_t getSwizzledColumn(uint32_t row, uint32_t col,
                                             uint32_t numCols) {
  return (row ^ col) % numCols;
}

template <uint32_t skew = 1, uint32_t baseRow = UINT32_MAX>
inline __device__ uint32_t getSkewCol(uint32_t row, uint32_t col,
                                      uint32_t numCols) {
  return (col + (row % baseRow)*skew) % numCols;
}


inline __device__ uint32_t swap(uint32_t& x, uint32_t & y) {
  uint32_t temp = x;
  x = y;
  y = temp;
  return temp;
}

} // end namespace flash::common
