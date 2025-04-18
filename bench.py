import math

import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load

# Load the CUDA kernel as a python module
minimal_attn = load(name='minimal_attn', sources=['main.cpp', 'flash.cu', 'flash_2d.cu'], extra_cuda_cflags=['-O2', '-std=c++20'])

# Use small model params, otherwise slower than manual attention. See caveats in README.
# batch_size = 1
# n_head = 1
# seq_len = 32
# head_embd = 32

batch_size = 64
n_head = 16
seq_len = 512
head_embd = 64

torch.manual_seed(0)
q = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
k = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
v = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()

# Our minimal flash attention aims to be faster than this by avoiding HBM read/writes of N^2 matrices.
def manual_attn(q, k, v):
    att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y


print('=== correctness check ===')
manual_result = manual_attn(q, k, v)
minimal_result = minimal_attn.forward(q, k, v)
igor_result = minimal_attn.forward_opt(q, k, v)
# if torch.allclose(minimal_result, manual_result, rtol=1e-2):
#     print('=== correctness check passed ===')
# else:
#     print('=== correctness check failed ===')
#     print(manual_result)
#     print(minimal_result)
#     exit(0)

if torch.allclose(igor_result, manual_result, rtol=1e-2, atol=1e-4):
    print('=== correctness check passed ===')
else:
    print('=== correctness check failed ===')
    print("reference")
    print(manual_result)
    print("igor")
    print(igor_result)
    exit(0)



print('=== profiling manual attention ===')

with torch.autograd.profiler.profile(use_device='cuda') as prof:
    manual_result = manual_attn(q, k, v)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print('=== profiling minimal flash attention === ')

with torch.autograd.profiler.profile(use_device='cuda') as prof:
    minimal_result = minimal_attn.forward(q, k, v)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

with torch.autograd.profiler.profile(use_device='cuda') as prof:
    igor_result = minimal_attn.forward_opt(q, k, v)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

# print('attn values sanity check:', torch.allclose(minimal_result, manual_result, rtol=0, atol=1e-02))
