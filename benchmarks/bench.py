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

# Load the CUDA kernel as a python module

# # Use small model params, otherwise slower than manual attention. See caveats in README.
batch_size = 8
num_heads = 12
seq_len = 1024
# seq_len = 12
head_embd = 64

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

# cold run the kernels before profiling
manual_result = manual_attn(q, k, v)
naive_result = pyflash.naive(q, k, v)
scalar_2d_result = pyflash.scalar2d(q, k, v)
warp_wmma_sync_result = pyflash.warp_wmma_sync(q, k, v)

with profiler.profile(use_device='cuda') as prof:
    O = manual_attn(q, k, v)
    print(prof.key_averages().table())
# print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=profilerRowLimit))

print('=== profiling minimal flash attention === ')

with torch.autograd.profiler.profile(use_device='cuda') as prof:
    O = torch.ops.pyflash.naive(q, k, v)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=5))

with torch.autograd.profiler.profile(use_device='cuda') as prof:
    O = torch.ops.pyflash.scalar2d(q, k, v)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=5))
