#pragma once
#include <cmath>
#include <stdexcept>
#include "cuda_constants.hpp"
#include "common.hpp"
#include "launch.hpp"

namespace flash {

constexpr int maxWarpsPerBlock = constants::maxBlockSize / constants::warpSize;

static inline int maxTileSizeForDeviceSharedMemory(int head_dim) {
  /*
    * We compute the tile size assuming the maximum use of shared memory.
    * This is done by solving a quadratic equation:
    * sh_size = 3*Bc*d + (Bc*Br) -> solve for Bc
    * Assume Bc = Br = x
    * x = (sqrt(9*d*d + 4*m) - 3*d)/2
    */
  int max_sram_size;
  cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
  int const c = max_sram_size / sizeof(float);
  int d = head_dim;
  const int tileSize = floor((sqrt(9. * d * d + 4. * c) - 3. * d) / 2.);
  return tileSize;
}

static uint sramSizeForTiles(int d, int Br, int Bc) {
  return (3*Bc*d + Br*Bc) * sizeof(float);
}

class KernelParameters {
public:
  KernelParameters() = default;
  virtual dim3 tileSize() = 0;
  virtual uint sramSize() = 0;
  virtual dim3 blockDim() = 0;
};

class NaiveKernelParameters : public KernelParameters {
  int d{};
public:
  NaiveKernelParameters(int head_dimension)
  : d(head_dimension) {}

  dim3 tileSize() {
    auto ts = std::min(maxTileSizeForDeviceSharedMemory(d),
                      constants::maxBlockSize);
    return dim3(ts, ts);
  }

  uint sramSize() {
    auto ts = tileSize();
    return sramSizeForTiles(d, ts.x, ts.y);
  }

  dim3 blockDim() {
    return dim3(tileSize().x, 1);
  }
};

class Scalar2DKernelParameters : public KernelParameters {
  int d{};
public:
  Scalar2DKernelParameters(int head_dimension)
  : d(head_dimension) {}

  dim3 tileSize() {
    auto ts = std::min(maxTileSizeForDeviceSharedMemory(d), maxWarpsPerBlock);
    return dim3(ts, ts);
  }

  uint sramSize() {
    auto ts = tileSize();
    return sramSizeForTiles(d, ts.x, ts.y);
  }

  dim3 blockDim() {
    auto ts = tileSize();
    return dim3(constants::warpSize, std::max(ts.x, ts.y));
  }
};

class WarpWMMASyncKernelParameters : public KernelParameters {
  int d{};
public:
  WarpWMMASyncKernelParameters(int head_dimension)
  : d(common::nextMultiple(head_dimension, constants::WMMA_N))
  {}

  dim3 tileSize() {
    return dim3(constants::WMMA_M, constants::WMMA_M);
  }
  uint sramSize() {
    auto ts = tileSize();
    return sramSizeForTiles(d, ts.x, ts.y);
  }
  dim3 blockDim() {
    return {constants::warpSize};
  }
};

class BlockWMMASyncKernelParameters : public KernelParameters {
  int d{};
public:
  BlockWMMASyncKernelParameters(int head_dimension)
  : d(common::nextMultiple(head_dimension, constants::WMMA_N))
  {}

  dim3 tileSize() {
    auto ts = std::min(maxTileSizeForDeviceSharedMemory(d), maxWarpsPerBlock);
    return dim3(ts, ts);
  }

  uint sramSize() {
    auto ts = tileSize();
    return sramSizeForTiles(d, ts.x, ts.y);
  }

  dim3 blockDim() {
    auto ts = tileSize();
    return dim3(constants::warpSize, max(ts.x, ts.y));
  }
};

class BlockWMMAAsyncKernelParameters : public KernelParameters {
  int d{};
public:
  BlockWMMAAsyncKernelParameters(int head_dimension)
  : d(common::nextMultiple(head_dimension, constants::WMMA_N))
  {}
  dim3 tileSize() {
    // x = Bc = Br
    // 3*x*d + x^2 + x = M
    // solve for x: x^2 + (3*d + 1)*x - M = 0
    // x = 1/2 * ( sqrt(9*d*d + 6*d + 4*M + 1) - 3*d - 1)
    int M;
    cudaDeviceGetAttribute(&M, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    M /= sizeof(float);
    auto ts = floor(0.5 * ( sqrt(9.*d*d + 6.*d + 4.*M + 1.) - 3.*d - 1.));

    // must be a multiple of WMMA_M
    auto ts_y = (int)(ts / constants::WMMA_M) * constants::WMMA_M;
    auto ts_x = (int)(ts / constants::WMMA_N) * constants::WMMA_N;

    return dim3(ts_x, ts_y);
  }
  uint sramSize() {
    auto ts = tileSize();
    auto Bc = ts.x;
    auto Br = ts.y;
    // 3*x*d + x^2 + x = M
    return sramSizeForTiles(d, Br, Bc) + Bc*sizeof(float);
  }

  dim3 blockDim() {
    auto ts = tileSize();
    // auto nWarps = max(ts.x, ts.y);
    auto nWarps = 6;
    nWarps = min(nWarps, maxWarpsPerBlock);
    return dim3(constants::warpSize, nWarps);
  }
};

class FlashKernelParametersFactory {
public:
  static std::unique_ptr<KernelParameters> create(int head_dimension, KernelType kernelType) {
    switch (kernelType) {
      case KernelType::naive1D:
        return std::make_unique<NaiveKernelParameters>(head_dimension);
      case KernelType::scalar2D:
        return std::make_unique<Scalar2DKernelParameters>(head_dimension);
      case KernelType::warp_wmma_sync:
        return std::make_unique<WarpWMMASyncKernelParameters>(head_dimension);
      case KernelType::block_wmma_sync:
        return std::make_unique<BlockWMMASyncKernelParameters>(head_dimension);
      case KernelType::block_wmma_async:
        return std::make_unique<BlockWMMAAsyncKernelParameters>(head_dimension);
      default: {
        std::ostringstream err;
        err << __FILE__ << "(" << __LINE__ << ") "
            << "error: unknown kernel type";
        throw std::invalid_argument(err.str());
      }
    }
  }
};

}
