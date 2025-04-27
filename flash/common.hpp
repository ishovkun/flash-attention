#include <cuda.h>

namespace flash::common {

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
    return __shfl_sync(UINT32_MAX, value, 0);
  }

} // end namespace flash::common
