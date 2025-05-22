# flash-attention

This is an implementation of the Flash attention V2 in raw CUDA without any libraries (e.g. Cutlass).
This started as fork of [flash-attention-minimal](https://github.com/tspeterkim/flash-attention-minimal),
and I was wondering how far I could get this (see the benchmark below).

The code is organized as multiple kernels, each representing a separate optimization step.
Therefore, it might serve as an educational source (see [Implemented Kernels](#implemented-kernels).

## Benchmark

This is the benchmark on the parameters of a transformer from GPT-3.
The benchmark has been measured on a single A100 GPU.
![scalability image](img/scalability.png "scalability")

## Build Instructions

- Install CUDA Toolkit (I used 12.8).
- Install pytorch (I used 2.6.0).
- Clone [flash-attention](https://github.com/ishovkun/flash-attention).

```bash
git clone https://github.com/ishovkun/flash-attention.git
cd flash-attention
mkdir build && cd build
```

- Determine the Compute Capability of your GPU:

```bash
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv | sed -n '2p' | sed 's/\.//g')
```

- Determine Pytorch path:

```bash
TORCH_PATH=$(python -c "import torch; print(torch.__path__[0])")
```

- Configure the build:

```bash
cmake .. -DTorch_DIR=${TORCH_PATH}/share/cmake/Torch \
        -DCMAKE_CUDA_ARCHITECTURES=${COMPUTE_CAP} \
        -DCMAKE_BUILD_TYPE=Release
```

- Compile the code:

```bash
make -j
```

## Run Instructions

- First, run the C++ code to make sure that the application built correctly. The runner runs the correctness tests.

```bash
./runner
```

- Next, run the Python benchmark:

```bash
python ../benchmarks/bench.py
```

## Implemented Kernels

- `flash_naive`: This code is an implementation from [flash-attention-minimal](https://github.com/tspeterkim/flash-attention-minimal). It does many things wrong, e.g. non-coalesced memory loads.
- `flash_2d`: This kernel uses 2D grid blocks; the memory loads are coalesced.
- `warp_wmma`: This kernel is taken from [flash-attention-minimal](https://github.com/tspeterkim/flash-attention-minimal).
  It uses cuda's wmma tensor core instructions and launches single-warp CTAs.
- `block_wmma`: This kernel builds on `warp_wmma` but uses multiple warps in a CTA.
- `scalar2d_row_tile`: This kernel is scalar but processes multiple rows of a single attention head instead of a full head as opposed to previous versions.
- `wmma_row_block`: This kernel is a block version of `scalar2d_row_tile` and uses wmma instructions.
- `kernel_mma`: This kernel is a block version of `scalar2d_row_tile` and uses mma (inline assembly) instructions.
- `kernel_mma_swizzle`: Same as previous version but uses skeweing (a form of swizzling) to improve memory access patterns.
- `kernel_mma_qreg`: This kernel uses the algorithm from FlashAttention2. It stores the Q tile in registers, thus minimizing the number of thread synchronizations. It also uses skewing and vectorized stores into the global memory.
- `kernel_mma_qreg_async`: This kernel builds on the previous version but uses async memory copies for K and V matrices.
