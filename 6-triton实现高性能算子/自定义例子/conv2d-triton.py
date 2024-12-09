import torch

import torch.nn as nn
from torch.nn import functional as F
from torch.utils.cpp_extension import load
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)
torch.backends.cudnn.benchmark = True
# turn off TF32 for higher accuracy
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

import triton

import triton.language as tl

######################################################################################################
# The formula:
# w.shape = (C, T)
# k.shape = (B, C, T)
# out.shape = (B, C, T)
# out[b][c][t] = sum_u{ w[c][(T-1)-(t-u)] * k[b][c][u] }
######################################################################################################


def RUN_FORMULA_VERY_SLOW(weight, k, B, C, T, eps):
    # this is the formula (very slow)
    out = torch.empty((B, C, T), device='cuda')
    for b in range(B):
        for c in range(C):
            for t in range(T):
                s = eps
                for u in range(0, t+1):
                    s += weight[c][(T-1)-(t-u)] * k[b][c][u]
                out[b][c][t] = s
    return out

@triton.jit
def RUN_TRITON(w, k, B, C, T, eps):

    return

# @torch.compile
def RUN_PYTORCH(weight, k, B, C, T, eps):
    # this shall equal the formula
    # return nn.Conv1d(nn.ZeroPad2d((T-1, 0, 0, 0))(k), w.unsqueeze(1), groups=C) + eps
    # m = nn.Conv1d(16, 33, 3, groups=C)
    # output = m(nn.ZeroPad2d((T-1, 0, 0, 0))(k))
    
    return F.conv1d(nn.ZeroPad2d((T-1, 0, 0, 0))(k), weight.unsqueeze(1),groups=C) + eps


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def CHECK_PYTORCH():
    B = 3
    C = 5
    T = 11
    eps = 0.1

    set_seed(42)
    weight = torch.rand(5, 11, requires_grad=True, device='cuda')
    k = torch.rand(3, 5, 11, requires_grad=True, device='cuda')

    r0 = RUN_FORMULA_VERY_SLOW(weight, k, 3, 5, 11, eps)
    r1 = RUN_PYTORCH(weight, k, 3, 5, 11, eps)

    print('--> pytorch correct =', torch.allclose(r0, r1))



if __name__ == "__main__":
    print('\n\nVerify pytorch...')
    CHECK_PYTORCH()

