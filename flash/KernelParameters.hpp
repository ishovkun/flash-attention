#pragma once
#include "common.hpp"
#include "cuda_constants.hpp"
#include "launch.hpp"
#include <cmath>
#include <stdexcept>

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
  return (3 * Bc * d + Br * Bc) * sizeof(float);
}

class KernelParameters {
public:
  KernelParameters() = default;
  virtual dim3 tileSize() = 0;
  virtual uint sramSize() = 0;
  virtual dim3 blockDim() = 0;
  virtual dim3 gridDim() = 0;
};

class NaiveKernelParameters : public KernelParameters {
  uint batchSize;
  uint numHeads;
  uint seqLen;
  uint headDim;

public:
  NaiveKernelParameters(int batchSize, int numHeads, int seqLen, int headDim)
      : batchSize(batchSize), numHeads(numHeads), seqLen(seqLen),
        headDim(headDim) {}

  dim3 tileSize() {
    auto ts = std::min(maxTileSizeForDeviceSharedMemory(headDim),
                       constants::maxBlockSize);
    return dim3(ts, ts);
  }
  uint sramSize() {
    auto ts = tileSize();
    return sramSizeForTiles(headDim, ts.x, ts.y);
  }
  dim3 blockDim() { return dim3(tileSize().x, 1); }
  dim3 gridDim() { return {batchSize, numHeads, 1}; }
};

class Scalar2DKernelParameters : public KernelParameters {
  uint batchSize;
  uint numHeads;
  uint seqLen;
  uint headDim;

public:
  Scalar2DKernelParameters(int batchSize, int numHeads, int seqLen, int headDim)
      : batchSize(batchSize), numHeads(numHeads), seqLen(seqLen),
        headDim(headDim) {}

  dim3 tileSize() {
    auto ts = std::min(maxTileSizeForDeviceSharedMemory(headDim), maxWarpsPerBlock);
    return dim3(ts, ts);
  }

  uint sramSize() {
    auto ts = tileSize();
    return sramSizeForTiles(headDim, ts.x, ts.y);
  }

  dim3 blockDim() {
    auto ts = tileSize();
    return dim3(constants::warpSize, std::max(ts.x, ts.y));
  }

  dim3 gridDim() { return {batchSize, numHeads, 1}; }
};

class Scalar2DRowTileKernelParameters : public KernelParameters {
  uint batchSize;
  uint numHeads;
  uint seqLen;
  uint headDim;

public:
  Scalar2DRowTileKernelParameters(int batchSize, int numHeads, int seqLen, int headDim)
  : batchSize(batchSize), numHeads(numHeads), seqLen(seqLen),
    headDim(headDim) {}

  dim3 tileSize() {
    auto ts = std::min(maxTileSizeForDeviceSharedMemory(headDim), maxWarpsPerBlock);
    return dim3(ts, ts);
  }

  uint sramSize() {
    auto ts = tileSize();
    return sramSizeForTiles(headDim, ts.x, ts.y);
  }

  dim3 blockDim() {
    auto ts = tileSize();
    auto warpsPerTile = ts.y;
    return dim3(constants::warpSize, warpsPerTile);
  }

  dim3 gridDim() {
    auto ts = tileSize();
    auto warpsPerTile = ts.y;
    uint blocksPerSequence = common::ceil_div(seqLen, warpsPerTile);
    return {batchSize, numHeads, blocksPerSequence};
  }
};

class WarpWMMASyncKernelParameters : public KernelParameters {
  uint batchSize;
  uint numHeads;
  uint seqLen;
  uint headDim;

public:
  WarpWMMASyncKernelParameters(int batchSize, int numHeads, int seqLen, int headDim)
      : batchSize(batchSize), numHeads(numHeads), seqLen(seqLen),
        headDim(common::nextMultiple(headDim, constants::WMMA_N)) {}

  dim3 tileSize() { return dim3(constants::WMMA_M, constants::WMMA_M); }

  uint sramSize() {
    auto ts = tileSize();
    return sramSizeForTiles(headDim, ts.x, ts.y);
  }
  dim3 blockDim() { return {constants::warpSize}; }

  dim3 gridDim() { return {batchSize, numHeads, 1}; }
};

class BlockWMMASyncKernelParameters : public KernelParameters {
  uint batchSize;
  uint numHeads;
  uint seqLen;
  uint headDim;

public:
  BlockWMMASyncKernelParameters(int batchSize, int numHeads, int seqLen, int headDim)
  : batchSize(batchSize), numHeads(numHeads), seqLen(seqLen),
    headDim(common::nextMultiple(headDim, constants::WMMA_N))
  {}

  dim3 tileSize() {
    auto ts = std::min(maxTileSizeForDeviceSharedMemory(headDim), maxWarpsPerBlock);
    return dim3(ts, ts);
  }

  uint sramSize() {
    auto ts = tileSize();
    return sramSizeForTiles(headDim, ts.x, ts.y);
  }

  dim3 blockDim() {
    auto ts = tileSize();
    return dim3(constants::warpSize, max(ts.x, ts.y));
  }

  dim3 gridDim() { return {batchSize, numHeads, 1}; }
};

class BlockWMMAAsyncKernelParameters : public KernelParameters {
  uint batchSize;
  uint numHeads;
  uint seqLen;
  uint headDim;

public:
  BlockWMMAAsyncKernelParameters(int batchSize, int numHeads, int seqLen, int headDim)
  : batchSize(batchSize), numHeads(numHeads), seqLen(seqLen),
    headDim(common::nextMultiple(headDim, constants::WMMA_N))
  {}

  dim3 tileSize() {
    // x = Bc = Br
    // 3*x*d + x^2 + x = M
    // solve for x: x^2 + (3*d + 1)*x - M = 0
    // x = 1/2 * ( sqrt(9*d*d + 6*d + 4*M + 1) - 3*d - 1)
    int M;
    cudaDeviceGetAttribute(&M, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    M /= sizeof(float);
    auto d = headDim;
    auto ts = floor(0.5 * (sqrt(9. * d * d + 6. * d + 4. * M + 1.) - 3. * d - 1.));

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
    return sramSizeForTiles(headDim, Br, Bc) + Bc * sizeof(float);
  }

  dim3 blockDim() {
    auto ts = tileSize();
    // auto nWarps = max(ts.x, ts.y);
    auto nWarps = 6;
    nWarps = min(nWarps, maxWarpsPerBlock);
    return dim3(constants::warpSize, nWarps);
  }

  dim3 gridDim() { return {batchSize, numHeads, 1}; }
};

class FlashKernelParametersFactory {
public:
  static std::unique_ptr<KernelParameters> create(int batchSize, int numHeads,
                                                  int seqLen, int headDim,
                                                  KernelType kernelType) {
    switch (kernelType) {
    case KernelType::naive1D:
      return std::make_unique<NaiveKernelParameters>(batchSize, numHeads,
                                                     seqLen, headDim);
    case KernelType::scalar2D:
      return std::make_unique<Scalar2DKernelParameters>(batchSize, numHeads,
                                                        seqLen, headDim);
    case KernelType::scalar2D_row_tile:
      return std::make_unique<Scalar2DRowTileKernelParameters>(
          batchSize, numHeads, seqLen, headDim);
    case KernelType::warp_wmma_sync:
      return std::make_unique<WarpWMMASyncKernelParameters>(batchSize, numHeads,
                                                            seqLen, headDim);
    case KernelType::block_wmma_sync:
      return std::make_unique<BlockWMMASyncKernelParameters>(
          batchSize, numHeads, seqLen, headDim);
    case KernelType::block_wmma_async:
      return std::make_unique<BlockWMMAAsyncKernelParameters>(
          batchSize, numHeads, seqLen, headDim);
    default: {
      std::ostringstream err;
      err << __FILE__ << "(" << __LINE__ << ") "
          << "error: unknown kernel type";
      throw std::invalid_argument(err.str());
    }
    }
  }
};

} // namespace flash
