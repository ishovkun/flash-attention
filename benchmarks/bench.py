import math

import torch
from torch.nn import functional as F
# from torch.utils.cpp_extension import load
from pathlib import Path

print("Pytorch version:", torch.__version__)
profilerRowLimit = 5

projectDir = Path(__file__).parent.parent.resolve()
pyflashPath = str(projectDir / "build/pyflash/libpyflash.so")
print("loading pyflash from", pyflashPath)
torch.ops.load_library(pyflashPath)

import torch.autograd.profiler as profiler
from torch.ops import pyflash

profiler_print_cuda_time_only = True

# GPT2
# batch_size = 8
# num_heads = 12
# seq_len = 1024
# head_embd = 64

#GPT3
batch_size = 4
num_heads = 96
seq_len = 2048
head_embd = 128

torch.manual_seed(0)
q = torch.randn(batch_size, num_heads, seq_len, head_embd).cuda()
k = torch.randn(batch_size, num_heads, seq_len, head_embd).cuda()
v = torch.randn(batch_size, num_heads, seq_len, head_embd).cuda()

# # Our minimal flash attention aims to be faster than this by avoiding HBM read/writes of N^2 matrices.
def manual_attn(q, k, v):
    att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y

# warmup run the kernels before profiling
manual_result = manual_attn(q, k, v)
naive_result = pyflash.naive(q, k, v)
scalar_2d_result = pyflash.scalar2d(q, k, v)
scalar_2d_row_tile_result = pyflash.scalar2d_row_tile(q, k, v)
warp_wmma_result = pyflash.warp_wmma(q, k, v)
block_wmma_result = pyflash.block_wmma(q, k, v)
block_wmma_async_result = pyflash.block_wmma_async(q, k, v)
block_wmma_async_result = pyflash.block_wmma_async(q, k, v)
mma_result = pyflash.mma(q, k, v)
mma_swizle_result = pyflash.mma_swizzle(q, k, v)

def profile_kernel(kernel, q, k, v, gpu_time_only):
    with profiler.profile(use_device='cuda') as prof:
        O = kernel(q, k, v)
    if not gpu_time_only:
        print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=profilerRowLimit))
    else:
        total_gpu_time_us = sum([item.self_device_time_total for item in prof.key_averages()])
        print("{}: {} ms".format(kernel.__name__, total_gpu_time_us / 1e3))

profile_kernel(manual_attn, q, k, v, gpu_time_only=profiler_print_cuda_time_only)
profile_kernel(pyflash.naive, q, k, v, gpu_time_only=profiler_print_cuda_time_only)
profile_kernel(pyflash.scalar2d, q, k, v, gpu_time_only=profiler_print_cuda_time_only)
profile_kernel(pyflash.scalar2d_row_tile, q, k, v, gpu_time_only=profiler_print_cuda_time_only)
profile_kernel(pyflash.warp_wmma, q, k, v, gpu_time_only=profiler_print_cuda_time_only)
profile_kernel(pyflash.block_wmma, q, k, v, gpu_time_only=profiler_print_cuda_time_only)
profile_kernel(pyflash.block_wmma_async, q, k, v, gpu_time_only=profiler_print_cuda_time_only)
profile_kernel(pyflash.block_wmma_row_block, q, k, v, gpu_time_only=profiler_print_cuda_time_only)
profile_kernel(pyflash.mma, q, k, v, gpu_time_only=profiler_print_cuda_time_only)
profile_kernel(pyflash.mma_swizzle, q, k, v, gpu_time_only=profiler_print_cuda_time_only)
