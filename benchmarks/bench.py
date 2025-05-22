import math
import torch
from torch.nn import functional as F
# torch_flash_v2 = F.scaled_dot_product_attention
from torch.nn.functional import scaled_dot_product_attention as torch_flash_v2
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
# naive_result = pyflash.naive(q, k, v)
# scalar_2d_result = pyflash.scalar2d(q, k, v)
# scalar_2d_row_tile_result = pyflash.scalar2d_row_tile(q, k, v)
# warp_wmma_result = pyflash.warp_wmma(q, k, v)
# block_wmma_result = pyflash.block_wmma(q, k, v)
# block_wmma_async_result = pyflash.block_wmma_async(q, k, v)
# block_wmma_async_result = pyflash.block_wmma_async(q, k, v)
# mma_result = pyflash.mma(q, k, v)
# mma_swizle_result = pyflash.mma_swizzle(q, k, v)
# mma_qreg_result = pyflash.mma_qreg(q, k, v)

def profile_kernel(kernel, q, k, v, gpu_time_only, use_cuda=True):
    with profiler.profile(use_cuda=use_cuda) as prof:
        O = kernel(q, k, v)
    if not gpu_time_only:
        print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=profilerRowLimit))
    else:
        total_gpu_time_us = sum([item.self_device_time_total for item in prof.key_averages()])
        print("{}: {} ms".format(kernel.__name__, total_gpu_time_us / 1e3))
        return total_gpu_time_us

# profile_kernel(pyflash.naive, q, k, v, gpu_time_only=profiler_print_cuda_time_only)
# profile_kernel(pyflash.scalar2d, q, k, v, gpu_time_only=profiler_print_cuda_time_only)
# profile_kernel(pyflash.scalar2d_row_tile, q, k, v, gpu_time_only=profiler_print_cuda_time_only)
# profile_kernel(pyflash.warp_wmma, q, k, v, gpu_time_only=profiler_print_cuda_time_only)
# profile_kernel(pyflash.block_wmma, q, k, v, gpu_time_only=profiler_print_cuda_time_only)
# profile_kernel(pyflash.block_wmma_async, q, k, v, gpu_time_only=profiler_print_cuda_time_only)
# profile_kernel(pyflash.block_wmma_row_block, q, k, v, gpu_time_only=profiler_print_cuda_time_only)
# profile_kernel(pyflash.mma, q, k, v, gpu_time_only=profiler_print_cuda_time_only)
# profile_kernel(pyflash.mma_swizzle, q, k, v, gpu_time_only=profiler_print_cuda_time_only)
# profile_kernel(pyflash.mma_qreg, q, k, v, gpu_time_only=profiler_print_cuda_time_only)
# profile_kernel(pyflash.mma_qreg_vld, q, k, v, gpu_time_only=profiler_print_cuda_time_only)
# profile_kernel(manual_attn, q, k, v, gpu_time_only=profiler_print_cuda_time_only)
# # torch_flash_v2 = F.scaled_dot_product_attention
# profile_kernel(torch_flash_v2, q, k, v, gpu_time_only=profiler_print_cuda_time_only)

def insert(data, kernel, time):
    name = kernel.__name__
    if name not in data:
        data[name] = []
    data[name].append(time)

print("\n========================")
print("Scalability")
data = {"Sequence Length": [512, 1024, 1024*2, 1024*4, 1024*8, 1024*16]}

for seq_len in data["Sequence Length"]:
    q = torch.randn(batch_size, num_heads, seq_len, head_embd).cuda()
    k = torch.randn(batch_size, num_heads, seq_len, head_embd).cuda()
    v = torch.randn(batch_size, num_heads, seq_len, head_embd).cuda()
    print("\nseq_len", seq_len)
    try:
        time = profile_kernel(manual_attn, q, k, v, gpu_time_only=profiler_print_cuda_time_only)
        insert(data, manual_attn, time)
    except torch.OutOfMemoryError as e:
        insert(data, manual_attn, 0)
        print("Out of memory: ", e)
    time = profile_kernel(pyflash.mma_qreg_vld, q, k, v, gpu_time_only=profiler_print_cuda_time_only)
    insert(data, pyflash.mma_qreg_vld, time)
    time = profile_kernel(pyflash.mma_qreg_async, q, k, v, gpu_time_only=profiler_print_cuda_time_only)
    insert(data, pyflash.mma_qreg_async, time)
    time = profile_kernel(torch_flash_v2, q, k, v, gpu_time_only=profiler_print_cuda_time_only)
    insert(data, torch_flash_v2, time)

print(data)

# for comparison, run the manual attention kernel on CPU
# qc = q.cpu()
# kc = k.cpu()
# vc = v.cpu()
# profile_kernel(manual_attn, qc, kc, vc, gpu_time_only=False, use_cuda=False)
