#pragma once
#include "common.hpp"
#include "launch.hpp"
#include "mma.hpp"
#include "wmma.hpp"
#include <cmath>
#include <stdexcept>

namespace flash {

constexpr int maxWarpsPerBlock = common::maxBlockSize / common::warpSize;

static inline uint32_t getDefaultMaxSRAM() {
  int max_sram_size;
  cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
  return max_sram_size;
}

static inline int maxTileSizeForSRAM(int head_dim, int sram) {
  /*
   * We compute the tile size assuming the maximum use of shared memory.
   * This is done by solving a quadratic equation:
   * sh_size = 3*Bc*d + (Bc*Br) -> solve for Bc
   * Assume Bc = Br = x
   * x = (sqrt(9*d*d + 4*m) - 3*d)/2
   */
  int const c = sram / sizeof(float);
  int d = head_dim;
  const int tileSize = floor((sqrt(9. * d * d + 4. * c) - 3. * d) / 2.);
  return tileSize;
}

static inline int maxTileSizeForDeviceSharedMemory(int head_dim) {
  /*
   * We compute the tile size assuming the maximum use of shared memory.
   * This is done by solving a quadratic equation:
   * sh_size = 3*Bc*d + (Bc*Br) -> solve for Bc
   * Assume Bc = Br = x
   * x = (sqrt(9*d*d + 4*m) - 3*d)/2
   */
  return maxTileSizeForSRAM(head_dim, getDefaultMaxSRAM());
}

static inline int maxBlockColumnsForBlockRows(int blockRows, int head_dim,
                                              int sram) {
  auto const S = sram / sizeof(float);
  auto const d = head_dim;
  auto const Br = blockRows;
  /*
   * S = Br*d + 2*Bc*d + (Bc*Br) -> solve for Bc
   */
  auto const Bc = (S - Br * d) / (Br + 2 * d);
  return Bc;
}

static uint sramSizeForTiles(int d, dim3 tileSize) {
  auto const Br = tileSize.y;
  auto const Bc = tileSize.x;
  return (Br * d + 2 * Bc * d + Br * Bc) * sizeof(float);
}

class KernelParametersBase {
public:
  KernelParametersBase() = default;
  virtual dim3 tileSize() = 0;
  virtual uint sramSize() = 0;
  virtual dim3 blockDim() = 0;
  virtual dim3 gridDim() = 0;
  virtual ~KernelParametersBase() = default;
};

class NaiveParameters : public KernelParametersBase {
  uint batchSize;
  uint numHeads;
  uint seqLen;
  uint headDim;

public:
  NaiveParameters(int batchSize, int numHeads, int seqLen, int headDim)
      : batchSize(batchSize), numHeads(numHeads), seqLen(seqLen),
        headDim(headDim) {}

  dim3 tileSize() {
    auto ts = std::min(maxTileSizeForDeviceSharedMemory(headDim),
                       common::maxBlockSize);
    return dim3(ts, ts);
  }
  uint sramSize() { return sramSizeForTiles(headDim, tileSize()); }
  dim3 blockDim() { return dim3(tileSize().x, 1); }
  dim3 gridDim() { return {batchSize, numHeads, 1}; }
};

class Scalar2DParameters : public KernelParametersBase {
  uint batchSize;
  uint numHeads;
  uint seqLen;
  uint headDim;

public:
  Scalar2DParameters(int batchSize, int numHeads, int seqLen, int headDim)
      : batchSize(batchSize), numHeads(numHeads), seqLen(seqLen),
        headDim(headDim) {}

  dim3 tileSize() {
    auto ts = std::min(maxTileSizeForDeviceSharedMemory(headDim), maxWarpsPerBlock);
    return dim3(ts, ts);
  }

  uint sramSize() { return sramSizeForTiles(headDim, tileSize()); }

  dim3 blockDim() {
    auto ts = tileSize();
    return dim3(common::warpSize, std::max(ts.x, ts.y));
  }

  dim3 gridDim() { return {batchSize, numHeads, 1}; }
};

class Scalar2DRowTileParameters : public KernelParametersBase {
  uint batchSize;
  uint numHeads;
  uint seqLen;
  uint headDim;

public:
  Scalar2DRowTileParameters(int batchSize, int numHeads, int seqLen,
                                  int headDim)
      : batchSize(batchSize), numHeads(numHeads), seqLen(seqLen),
        headDim(headDim) {}

  dim3 tileSize() {
    auto ts = std::min(maxTileSizeForDeviceSharedMemory(headDim), maxWarpsPerBlock);
    return dim3(ts, ts);
  }

  uint sramSize() { return sramSizeForTiles(headDim, tileSize()); }

  dim3 blockDim() {
    auto ts = tileSize();
    auto warpsPerTile = ts.y;
    return dim3(common::warpSize, warpsPerTile);
  }

  dim3 gridDim() {
    auto ts = tileSize();
    auto warpsPerTile = ts.y;
    uint blocksPerSequence = common::ceil_div(seqLen, warpsPerTile);
    return {batchSize, numHeads, blocksPerSequence};
  }
};

class WarpWMMAParameters : public KernelParametersBase {
  uint batchSize;
  uint numHeads;
  uint seqLen;
  uint headDim;

public:
  WarpWMMAParameters(int batchSize, int numHeads, int seqLen, int headDim)
      : batchSize(batchSize), numHeads(numHeads), seqLen(seqLen),
        headDim(common::nextMultiple(headDim, wmma::WMMA_N)) {}

  dim3 tileSize() { return dim3(wmma::WMMA_M, wmma::WMMA_M); }

  uint sramSize() {
    auto ts = tileSize();
    return sramSizeForTiles(headDim, tileSize());
  }
  dim3 blockDim() { return {common::warpSize}; }

  dim3 gridDim() { return {batchSize, numHeads, 1}; }
};

class BlockWMMAParameters : public KernelParametersBase {
  uint batchSize;
  uint numHeads;
  uint seqLen;
  uint headDim;

public:
  BlockWMMAParameters(int batchSize, int numHeads, int seqLen,
                                int headDim)
      : batchSize(batchSize), numHeads(numHeads), seqLen(seqLen),
        headDim(common::nextMultiple(headDim, wmma::WMMA_N)) {}

  dim3 tileSize() {
    auto ts =
        std::min(maxTileSizeForDeviceSharedMemory(headDim), maxWarpsPerBlock);
    return dim3(ts, ts);
  }

  uint sramSize() {
    auto ts = tileSize();
    return sramSizeForTiles(headDim, tileSize());
  }

  dim3 blockDim() {
    auto ts = tileSize();
    return dim3(common::warpSize, max(ts.x, ts.y));
  }

  dim3 gridDim() { return {batchSize, numHeads, 1}; }
};

class WMMARowBlockParameters : public KernelParametersBase {
  uint batchSize;
  uint numHeads;
  uint seqLen;
  uint headDim;

public:
  WMMARowBlockParameters(int batchSize, int numHeads, int seqLen,
                                   int headDim)
      : batchSize(batchSize), numHeads(numHeads), seqLen(seqLen),
        headDim(common::nextMultiple(headDim, wmma::WMMA_N)) {}

  dim3 tileSize() {
    auto ts =
        std::min(maxTileSizeForDeviceSharedMemory(headDim), maxWarpsPerBlock);
    return dim3(ts, ts);
  }

  uint sramSize() {
    auto ts = tileSize();
    return sramSizeForTiles(headDim, tileSize());
  }

  dim3 blockDim() {
    auto warpsPerTile = min(12, tileSize().y);
    // auto warpsPerTile = 1;
    return dim3(common::warpSize, warpsPerTile);
  }

  dim3 gridDim() {
    uint blocksPerHead = common::ceil_div(seqLen, tileSize().y);
    return {batchSize, numHeads, blocksPerHead};
  }
};

class MMAParameters : public KernelParametersBase {
  uint batchSize;
  uint numHeads;
  uint seqLen;
  uint headDim;

public:
  MMAParameters(int batchSize, int numHeads, int seqLen, int headDim)
      : batchSize(batchSize), numHeads(numHeads), seqLen(seqLen),
        headDim(common::nextMultiple(headDim, mma::Tile::N)) {}

  static constexpr uint32_t rowsPerBlock() { return 2 * mma::Tile::M; }
  static constexpr uint32_t warpsPerBlock() { return 12; }
  dim3 tileSize() {
    // Ampere: Up to 160KB of shared memory
    constexpr auto Br = rowsPerBlock();
    auto Bc = maxBlockColumnsForBlockRows(Br, headDim, getDefaultMaxSRAM());
    Bc = common::prevMultiple(Bc, mma::Tile::N);
    if (Bc < 2 * mma::Tile::N) {
      // auto S = 160 * 1024 / sizeof(float);
      // auto S = (int)(1.2 * 48 * 1024 / sizeof(float));
      auto S = (int)(48 * 1024 / sizeof(float));
      auto const d = headDim;
      Bc = (S - Br * d) / (Br + 2 * d);
      Bc = common::prevMultiple(Bc, mma::Tile::N);
    }
    // Bc = min(Br, Bc);
    Bc = max(Br, Bc);
    return dim3(Bc, Br);
  }

  uint sramSize() {
    auto ts = tileSize();
    return sramSizeForTiles(headDim, tileSize());
  }

  dim3 blockDim() {
    auto warpsPerTile = warpsPerBlock();
    return dim3(common::warpSize, warpsPerTile);
  }

  dim3 gridDim() {
    uint blocksPerHead = common::ceil_div(seqLen, tileSize().y);
    return {batchSize, numHeads, blocksPerHead};
  }
};

class BlockWMMAAsyncParameters : public KernelParametersBase {
  uint batchSize;
  uint numHeads;
  uint seqLen;
  uint headDim;

public:
  BlockWMMAAsyncParameters(int batchSize, int numHeads, int seqLen,
                                 int headDim)
      : batchSize(batchSize), numHeads(numHeads), seqLen(seqLen),
        headDim(common::nextMultiple(headDim, wmma::WMMA_N)) {}

  dim3 tileSize() {
    // x = Bc = Br
    // 3*x*d + x^2 + x = M
    // solve for x: x^2 + (3*d + 1)*x - M = 0
    // x = 1/2 * ( sqrt(9*d*d + 6*d + 4*M + 1) - 3*d - 1)
    int M = getDefaultMaxSRAM();
    M /= sizeof(float);
    auto d = headDim;
    auto ts = floor(0.5 * (sqrt(9. * d * d + 6. * d + 4. * M + 1.) - 3. * d - 1.));

    // must be a multiple of WMMA_M
    auto ts_y = (int)(ts / wmma::WMMA_M) * wmma::WMMA_M;
    auto ts_x = (int)(ts / wmma::WMMA_N) * wmma::WMMA_N;

    return dim3(ts_x, ts_y);
  }
  uint sramSize() {
    // 3*x*d + x^2 + x = M
    auto ts = tileSize();
    return sramSizeForTiles(headDim, ts) + ts.y * sizeof(float);
  }

  dim3 blockDim() {
    auto ts = tileSize();
    auto nWarps = 6;
    nWarps = min(nWarps, maxWarpsPerBlock);
    return dim3(common::warpSize, nWarps);
  }

  dim3 gridDim() { return {batchSize, numHeads, 1}; }
};

class MMAQregParameters : public KernelParametersBase {
  uint batchSize, numHeads, seqLen, headDim;
 public:
  MMAQregParameters(int batchSize, int numHeads, int seqLen, int headDim)
    : batchSize(batchSize), numHeads(numHeads), seqLen(seqLen),
      headDim(common::nextMultiple(headDim, 32))
  {}

  static constexpr uint32_t rowsPerTileQ() { return 8 * mma::Tile::M; } // 6 also works well
  static constexpr uint32_t rowsPerTileK() { return rowsPerTileQ() / 2; }
  static constexpr uint32_t warpsPerBlock() { return rowsPerTileQ() / mma::Tile::M; }
  dim3 tileSize() override { return dim3(rowsPerTileK(), rowsPerTileQ(), headDim); }
  uint32_t sramSize() override {
    auto tile = tileSize();
    auto Br = tile.y;
    auto Bc = tile.x;
    return sizeof(float)*(max(Br, 2*Bc)*headDim + Br*Bc);
  }
  dim3 blockDim() override { return dim3(common::warpSize, warpsPerBlock()); }
  dim3 gridDim() override {
    uint blocksPerHead = common::ceil_div(seqLen, tileSize().y);
    return {batchSize, numHeads, blocksPerHead};
  }
};

class KernelParametersFactory {
public:
  static std::unique_ptr<KernelParametersBase> create(int batchSize, int numHeads, int seqLen, int headDim, KernelType kernelType) {
    switch (kernelType) {
    case KernelType::naive1D:
      return std::make_unique<NaiveParameters>(batchSize, numHeads, seqLen, headDim);
    case KernelType::scalar2D:
      return std::make_unique<Scalar2DParameters>(batchSize, numHeads, seqLen, headDim);
    case KernelType::scalar2D_row_tile:
      return std::make_unique<Scalar2DRowTileParameters>(
          batchSize, numHeads, seqLen, headDim);
    case KernelType::warp_wmma:
      return std::make_unique<WarpWMMAParameters>(batchSize, numHeads, seqLen, headDim);
    case KernelType::block_wmma:
      return std::make_unique<BlockWMMAParameters>(
          batchSize, numHeads, seqLen, headDim);
    case KernelType::wmma_row_block:
      return std::make_unique<WMMARowBlockParameters>(
          batchSize, numHeads, seqLen, headDim);
    case KernelType::mma:
      return std::make_unique<MMAParameters>(batchSize, numHeads, seqLen, headDim);
    case KernelType::mma_swizzle:
      return std::make_unique<MMAParameters>(batchSize, numHeads, seqLen, headDim);
    case KernelType::mma_qreg:
      return std::make_unique<MMAQregParameters>(batchSize, numHeads, seqLen, headDim);
    case KernelType::mma_qreg_f32x4load:
      return std::make_unique<MMAQregParameters>(batchSize, numHeads, seqLen, headDim);
    case KernelType::block_wmma_async:
      return std::make_unique<BlockWMMAAsyncParameters>( batchSize, numHeads, seqLen, headDim);
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
