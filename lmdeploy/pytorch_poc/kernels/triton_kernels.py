# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from datetime import datetime

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton import Config
from triton.ops.matmul_perf_model import (early_config_prune,
                                          estimate_matmul_time)

# from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time

# from rsm_norm_w8a8 import linear_dynamic_quant_triton_op_fast, rms_norm_dynamic_quant


def per_channel_quant2(x, n_bits, eps, dtype):
    assert x.ndim == 2
    # x = x.to(torch.float32)
    x_absmax = x.view(x.shape[0], -1).abs().max(dim=1)[0]  #.clamp(min=eps)
    q_max = 2**(n_bits - 1) - 1
    q_min = -2**(n_bits - 1)
    scale = x_absmax / (2**(n_bits - 1) - 1)
    x_q = torch.round(x.float() / scale[:, None]).clamp(q_min, q_max).to(dtype)
    return x_q, scale


def per_channel_quant(x, n_bits, eps, dtype):
    assert x.ndim == 2
    x = x.to(torch.float32)
    x_absmax = x.view(x.shape[0],
                      -1).abs().max(dim=1, keepdim=True)[0]  #.clamp(min=eps)
    q_max = 2**(n_bits - 1) - 1
    q_min = -2**(n_bits - 1)
    scale = x_absmax / (2**(n_bits - 1) - 1)
    x_q = torch.round(x / scale).clamp(q_min, q_max).to(dtype)
    return x_q, scale


def get_configs_matmul(with_split_k=True):
    cfgs = [
        # basic configs for compute-bound matmuls
        # triton.Config(
        #     {
        #         'BLOCK_M': 128,
        #         'BLOCK_N': 256,
        #         'BLOCK_K': 32,
        #         'SPLIT_K': 1
        #     },
        #     num_stages=3,
        #     num_warps=8),
        # triton.Config(
        #     {
        #         'BLOCK_M': 256,
        #         'BLOCK_N': 128,
        #         'BLOCK_K': 32,
        #         'SPLIT_K': 1
        #     },
        #     num_stages=3,
        #     num_warps=8),
        # triton.Config(
        #     {
        #         'BLOCK_M': 256,
        #         'BLOCK_N': 64,
        #         'BLOCK_K': 32,
        #         'SPLIT_K': 1
        #     },
        #     num_stages=4,
        #     num_warps=4),
        # triton.Config(
        #     {
        #         'BLOCK_M': 64,
        #         'BLOCK_N': 256,
        #         'BLOCK_K': 32,
        #         'SPLIT_K': 1
        #     },
        #     num_stages=4,
        #     num_warps=4),
        # triton.Config(
        #     {
        #         'BLOCK_M': 128,
        #         'BLOCK_N': 128,
        #         'BLOCK_K': 32,
        #         'SPLIT_K': 1
        #     },
        #     num_stages=4,
        #     num_warps=4),
        # triton.Config(
        #     {
        #         'BLOCK_M': 128,
        #         'BLOCK_N': 64,
        #         'BLOCK_K': 32,
        #         'SPLIT_K': 1
        #     },
        #     num_stages=4,
        #     num_warps=4),
        # triton.Config(
        #     {
        #         'BLOCK_M': 64,
        #         'BLOCK_N': 128,
        #         'BLOCK_K': 32,
        #         'SPLIT_K': 1
        #     },
        #     num_stages=4,
        #     num_warps=4),
        # triton.Config(
        #     {
        #         'BLOCK_M': 128,
        #         'BLOCK_N': 32,
        #         'BLOCK_K': 32,
        #         'SPLIT_K': 1
        #     },
        #     num_stages=4,
        #     num_warps=4),
        # triton.Config(
        #     {
        #         'BLOCK_M': 64,
        #         'BLOCK_N': 32,
        #         'BLOCK_K': 32,
        #         'SPLIT_K': 1
        #     },
        #     num_stages=5,
        #     num_warps=2),
        # # good for int8
        # triton.Config(
        #     {
        #         'BLOCK_M': 128,
        #         'BLOCK_N': 256,
        #         'BLOCK_K': 128,
        #         'SPLIT_K': 1
        #     },
        #     num_stages=3,
        #     num_warps=8),
        # triton.Config(
        #     {
        #         'BLOCK_M': 256,
        #         'BLOCK_N': 128,
        #         'BLOCK_K': 128,
        #         'SPLIT_K': 1
        #     },
        #     num_stages=3,
        #     num_warps=8),
        # triton.Config(
        #     {
        #         'BLOCK_M': 256,
        #         'BLOCK_N': 64,
        #         'BLOCK_K': 128,
        #         'SPLIT_K': 1
        #     },
        #     num_stages=4,
        #     num_warps=4),
        # triton.Config(
        #     {
        #         'BLOCK_M': 64,
        #         'BLOCK_N': 256,
        #         'BLOCK_K': 128,
        #         'SPLIT_K': 1
        #     },
        #     num_stages=4,
        #     num_warps=4),
        # triton.Config(
        #     {
        #         'BLOCK_M': 128,
        #         'BLOCK_N': 128,
        #         'BLOCK_K': 128,
        #         'SPLIT_K': 1
        #     },
        #     num_stages=4,
        #     num_warps=4),
        # triton.Config(
        #     {
        #         'BLOCK_M': 128,
        #         'BLOCK_N': 64,
        #         'BLOCK_K': 64,
        #         'SPLIT_K': 1
        #     },
        #     num_stages=4,
        #     num_warps=4),
        # triton.Config(
        #     {
        #         'BLOCK_M': 64,
        #         'BLOCK_N': 128,
        #         'BLOCK_K': 64,
        #         'SPLIT_K': 1
        #     },
        #     num_stages=4,
        #     num_warps=4),
        # triton.Config(
        #     {
        #         'BLOCK_M': 128,
        #         'BLOCK_N': 32,
        #         'BLOCK_K': 64,
        #         'SPLIT_K': 1
        #     },
        #     num_stages=4,
        #     num_warps=4),
        # triton.Config(
        #     {
        #         'BLOCK_M': 64,
        #         'BLOCK_N': 32,
        #         'BLOCK_K': 64,
        #         'SPLIT_K': 1
        #     },
        #     num_stages=5,
        #     num_warps=2),
        # triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=3, num_warps=4),
        # triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=3, num_warps=4),
        # triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=3, num_warps=4),
        # triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        # triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=3, num_warps=4),
        # triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=4, num_warps=4),

        # triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 256, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=4, num_warps=4),

        # triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=3, num_warps=4),
        # triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=3, num_warps=4),
        # triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=3, num_warps=4),

        # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        # triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=5, num_warps=2),
        triton.Config(
            {
                'BLOCK_M': 32,
                'BLOCK_N': 64,
                'BLOCK_K': 32,
                'SPLIT_K': 1
            },
            num_stages=5,
            num_warps=2),
    ]
    if not with_split_k:
        for cfg in cfgs:
            del cfg.kwargs['SPLIT_K']
    return cfgs


def get_configs_block():
    configs = []
    for block_s in range(10, 16):
        block = int(2**block_s)
        num_warps = min(max(block // 256, 1), 8)
        for num_stages in [2, 3, 4, 5, 6]:
            configs.append(
                triton.Config({'BLOCK': block},
                              num_stages=num_stages,
                              num_warps=num_warps))
    return configs


@triton.autotune(configs=get_configs_block(), key=['MN'])
@triton.heuristics({
    'EVEN_BLOCK': lambda args: args['MN'] % args['BLOCK'] == 0,
})
@triton.jit
def _silu_elem_mul(
    x1_ptr,
    x2_ptr,
    y_ptr,
    MN,
    BLOCK: tl.constexpr,
    EVEN_BLOCK: tl.constexpr,
):
    # matrix multiplication
    pid = tl.program_id(0)
    cols = pid * BLOCK + tl.arange(0, BLOCK)
    if EVEN_BLOCK:
        x1 = tl.load(x1_ptr + cols)
        x2 = tl.load(x2_ptr + cols)
    else:
        mask = cols < MN
        x1 = tl.load(x1_ptr + cols, mask=mask, other=0.)
        x2 = tl.load(x2_ptr + cols, mask=mask, other=0.)
    y = (silu(x1.to(tl.float32)) * x2).to(y_ptr.dtype.element_ty)
    if EVEN_BLOCK:
        tl.store(y_ptr + cols, y)
    else:
        tl.store(y_ptr + cols, y, mask=mask)


def silu_elem_mul_tri(x1, x2):
    assert x1.shape == x2.shape

    M, N = x1.shape
    y = torch.empty((M * N, ), device=x1.device, dtype=x1.dtype)

    # launch kernel
    def grid(META):
        return (triton.cdiv(M * N, META['BLOCK']), )

    _silu_elem_mul[grid](x1.view(-1), x2.view(-1), y, M * N)

    return y.view(M, N)


@triton.autotune(configs=get_configs_matmul(), key=['M', 'N', 'K'])
@triton.heuristics({
    'EVEN_K':
    lambda args: args['K'] % (args['BLOCK_K'] * args['SPLIT_K']) == 0,
})
@triton.jit
def _linear_add_int8(x_q_ptr, w_q_ptr, x_s_ptr, w_s_ptr, residual_ptr, y_ptr,
                     M, N, K, stride_xm, stride_xk, stride_wk, stride_wn,
                     stride_ym, stride_yn, BLOCK_M: tl.constexpr,
                     BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                     GROUP_M: tl.constexpr, SPLIT_K: tl.constexpr,
                     EVEN_K: tl.constexpr):
    # matrix multiplication
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rxm = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rwn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)

    x_q_ptrs = x_q_ptr + (rxm[:, None] * stride_xm + rk[None, :] * stride_xk)
    w_q_ptrs = w_q_ptr + (rk[:, None] * stride_wk + rwn[None, :] * stride_wn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if EVEN_K:
            x_q = tl.load(x_q_ptrs)
            w_q = tl.load(w_q_ptrs)
        else:
            k_remaining = K - k * (BLOCK_K * SPLIT_K)
            _0 = tl.zeros((1, 1), dtype=x_q_ptrs.dtype.element_ty)
            x_q = tl.load(x_q_ptrs, mask=rk[None, :] < k_remaining, other=_0)
            w_q = tl.load(w_q_ptrs, mask=rk[:, None] < k_remaining, other=_0)
        # We accumulate along the K dimension.
        acc += tl.dot(x_q, w_q, out_dtype=tl.int32)
        x_q_ptrs += BLOCK_K * SPLIT_K * stride_xk
        w_q_ptrs += BLOCK_K * SPLIT_K * stride_wk
    x_s = tl.load(x_s_ptr + rxm, mask=rm < M, other=0.)
    w_s = tl.load(w_s_ptr + rwn, mask=rn < N, other=0.)
    y = acc * x_s[:, None] * w_s[None, :]
    y = y.to(residual_ptr.dtype.element_ty)

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    residual_ptrs = residual_ptr + (rm[:, None] * stride_ym +
                                    rn[None, :] * stride_yn)
    residual = tl.load(residual_ptrs, mask=mask, other=0.)
    y_ptrs = y_ptr + (rm[:, None] * stride_ym + rn[None, :] * stride_yn)
    if SPLIT_K == 1:
        tl.store(y_ptrs, y + residual, mask=mask)
    else:
        tl.atomic_add(y_ptrs, y + residual, mask=mask)


@triton.autotune(configs=get_configs_matmul(), key=['M', 'N', 'K'])
@triton.heuristics({
    'EVEN_K':
    lambda args: args['K'] % (args['BLOCK_K'] * args['SPLIT_K']) == 0,
})
@triton.jit
def _linear_int8(x_q_ptr, w_q_ptr, x_s_ptr, w_s_ptr, y_ptr, M, N, K, stride_xm,
                 stride_xk, stride_wk, stride_wn, stride_ym, stride_yn,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                 BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
                 SPLIT_K: tl.constexpr, EVEN_K: tl.constexpr):
    # matrix multiplication
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rxm = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rwn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)

    x_q_ptrs = x_q_ptr + (rxm[:, None] * stride_xm + rk[None, :] * stride_xk)
    w_q_ptrs = w_q_ptr + (rk[:, None] * stride_wk + rwn[None, :] * stride_wn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if EVEN_K:
            x_q = tl.load(x_q_ptrs)
            w_q = tl.load(w_q_ptrs)
        else:
            k_remaining = K - k * (BLOCK_K * SPLIT_K)
            _0 = tl.zeros((1, 1), dtype=x_q_ptrs.dtype.element_ty)
            x_q = tl.load(x1_q_ptrs, mask=rk[None, :] < k_remaining, other=_0)
            w_q = tl.load(w_q_ptrs, mask=rk[:, None] < k_remaining, other=_0)
        # We accumulate along the K dimension.
        acc += tl.dot(x_q, w_q, out_dtype=tl.int32)
        x_q_ptrs += BLOCK_K * SPLIT_K * stride_xk
        w_q_ptrs += BLOCK_K * SPLIT_K * stride_wk
    acc = acc.to(tl.float32)
    x_s = tl.load(x_s_ptr + rxm, mask=rm < M, other=0.)
    w_s = tl.load(w_s_ptr + rwn, mask=rn < N, other=0.)
    y = acc * x_s[:, None] * w_s[None, :]

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    y_ptrs = y_ptr + (rm[:, None] * stride_ym + rn[None, :] * stride_yn)
    if SPLIT_K == 1:
        tl.store(y_ptrs, y, mask=mask)
    else:
        tl.atomic_add(y_ptrs, y, mask=mask)


def linear_dynamic_quant_int8_tri(x_q,
                                  w_q,
                                  x_s,
                                  w_s,
                                  residual=None,
                                  output_dtype=torch.float16):
    assert x_q.shape[-1] == w_q.shape[-1]
    assert x_q.is_contiguous(), 'Matrix x_q must be contiguous'

    assert w_q.is_contiguous(), 'Matrix w_q.T must be contiguous'
    w_q = w_q.T
    assert w_q.stride(0) == 1

    M, K = x_q.shape
    _, N = w_q.shape
    if residual is not None:
        assert residual.shape == (M, N)
        assert residual.is_contiguous(), 'Matrix residual must be contiguous'
    y = torch.empty((M, N), device=x_q.device, dtype=output_dtype)

    # allocates output

    # launch kernel
    def grid(META):
        return (triton.cdiv(M, META['BLOCK_M']) *
                triton.cdiv(N, META['BLOCK_N']), META['SPLIT_K'])

    if residual is not None:
        _linear_add_int8[grid](x_q,
                               w_q,
                               x_s,
                               w_s,
                               residual,
                               y,
                               M,
                               N,
                               K,
                               x_q.stride(0),
                               x_q.stride(1),
                               w_q.stride(0),
                               w_q.stride(1),
                               y.stride(0),
                               y.stride(1),
                               GROUP_M=8)
    else:
        _linear_int8[grid](x_q,
                           w_q,
                           x_s,
                           w_s,
                           y,
                           M,
                           N,
                           K,
                           x_q.stride(0),
                           x_q.stride(1),
                           w_q.stride(0),
                           w_q.stride(1),
                           y.stride(0),
                           y.stride(1),
                           GROUP_M=8)
    return y


def get_configs_io_bound3():
    configs = []
    for num_stages in [2, 3, 4, 5, 6]:
        for block_m in [16, 32]:
            for block_k in [32, 64]:
                for block_n in [32, 64, 128, 256]:
                    num_warps = 2 if block_n <= 64 else 4
                    configs.append(
                        Config(
                            {
                                'BLOCK_M': block_m,
                                'BLOCK_N': block_n,
                                'BLOCK_K': block_k,
                                'SPLIT_K': 1
                            },
                            num_stages=num_stages,
                            num_warps=num_warps))
                    # split_k
                    # for split_k in [2, 4, 8, 16]:
                    #     configs.append(Config({'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 'SPLIT_K': split_k},
                    #                           num_stages=num_stages, num_warps=num_warps, pre_hook=init_to_zero('C')))
    return configs


@triton.autotune(
    configs=[
        # basic configs for compute-bound matmuls
        Config({
            'BLOCK_M': 128,
            'BLOCK_N': 256,
            'BLOCK_K': 32,
            'SPLIT_K': 1
        },
               num_stages=3,
               num_warps=8),
        Config({
            'BLOCK_M': 256,
            'BLOCK_N': 128,
            'BLOCK_K': 32,
            'SPLIT_K': 1
        },
               num_stages=3,
               num_warps=8),
        Config({
            'BLOCK_M': 256,
            'BLOCK_N': 64,
            'BLOCK_K': 32,
            'SPLIT_K': 1
        },
               num_stages=4,
               num_warps=4),
        Config({
            'BLOCK_M': 64,
            'BLOCK_N': 256,
            'BLOCK_K': 32,
            'SPLIT_K': 1
        },
               num_stages=4,
               num_warps=4),
        Config({
            'BLOCK_M': 128,
            'BLOCK_N': 128,
            'BLOCK_K': 32,
            'SPLIT_K': 1
        },
               num_stages=4,
               num_warps=4),
        Config({
            'BLOCK_M': 128,
            'BLOCK_N': 64,
            'BLOCK_K': 32,
            'SPLIT_K': 1
        },
               num_stages=4,
               num_warps=4),
        Config({
            'BLOCK_M': 64,
            'BLOCK_N': 128,
            'BLOCK_K': 32,
            'SPLIT_K': 1
        },
               num_stages=4,
               num_warps=4),
        Config({
            'BLOCK_M': 128,
            'BLOCK_N': 32,
            'BLOCK_K': 32,
            'SPLIT_K': 1
        },
               num_stages=4,
               num_warps=4),
        Config({
            'BLOCK_M': 64,
            'BLOCK_N': 32,
            'BLOCK_K': 32,
            'SPLIT_K': 1
        },
               num_stages=5,
               num_warps=2),
        # good for int8
        Config({
            'BLOCK_M': 128,
            'BLOCK_N': 256,
            'BLOCK_K': 128,
            'SPLIT_K': 1
        },
               num_stages=3,
               num_warps=8),
        Config({
            'BLOCK_M': 256,
            'BLOCK_N': 128,
            'BLOCK_K': 128,
            'SPLIT_K': 1
        },
               num_stages=3,
               num_warps=8),
        Config({
            'BLOCK_M': 256,
            'BLOCK_N': 64,
            'BLOCK_K': 128,
            'SPLIT_K': 1
        },
               num_stages=4,
               num_warps=4),
        Config({
            'BLOCK_M': 64,
            'BLOCK_N': 256,
            'BLOCK_K': 128,
            'SPLIT_K': 1
        },
               num_stages=4,
               num_warps=4),
        Config({
            'BLOCK_M': 128,
            'BLOCK_N': 128,
            'BLOCK_K': 128,
            'SPLIT_K': 1
        },
               num_stages=4,
               num_warps=4),
        Config({
            'BLOCK_M': 128,
            'BLOCK_N': 64,
            'BLOCK_K': 64,
            'SPLIT_K': 1
        },
               num_stages=4,
               num_warps=4),
        Config({
            'BLOCK_M': 64,
            'BLOCK_N': 128,
            'BLOCK_K': 64,
            'SPLIT_K': 1
        },
               num_stages=4,
               num_warps=4),
        Config({
            'BLOCK_M': 128,
            'BLOCK_N': 32,
            'BLOCK_K': 64,
            'SPLIT_K': 1
        },
               num_stages=4,
               num_warps=4),
        Config({
            'BLOCK_M': 64,
            'BLOCK_N': 32,
            'BLOCK_K': 64,
            'SPLIT_K': 1
        },
               num_stages=5,
               num_warps=2),
        triton.Config(
            {
                'BLOCK_M': 32,
                'BLOCK_N': 64,
                'BLOCK_K': 128,
                'SPLIT_K': 1
            },
            num_stages=3,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_M': 16,
                'BLOCK_N': 32,
                'BLOCK_K': 128,
                'SPLIT_K': 1
            },
            num_stages=3,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_M': 16,
                'BLOCK_N': 32,
                'BLOCK_K': 64,
                'SPLIT_K': 1
            },
            num_stages=3,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_M': 16,
                'BLOCK_N': 64,
                'BLOCK_K': 128,
                'SPLIT_K': 1
            },
            num_stages=3,
            num_warps=8),
        triton.Config(
            {
                'BLOCK_M': 16,
                'BLOCK_N': 64,
                'BLOCK_K': 128,
                'SPLIT_K': 1
            },
            num_stages=3,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_M': 16,
                'BLOCK_N': 64,
                'BLOCK_K': 128,
                'SPLIT_K': 1
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_M': 16,
                'BLOCK_N': 128,
                'BLOCK_K': 128,
                'SPLIT_K': 1
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_M': 16,
                'BLOCK_N': 128,
                'BLOCK_K': 256,
                'SPLIT_K': 1
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_M': 32,
                'BLOCK_N': 128,
                'BLOCK_K': 128,
                'SPLIT_K': 1
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_M': 64,
                'BLOCK_N': 128,
                'BLOCK_K': 128,
                'SPLIT_K': 1
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_M': 32,
                'BLOCK_N': 64,
                'BLOCK_K': 128,
                'SPLIT_K': 1
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_M': 64,
                'BLOCK_N': 64,
                'BLOCK_K': 128,
                'SPLIT_K': 1
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_M': 16,
                'BLOCK_N': 128,
                'BLOCK_K': 64,
                'SPLIT_K': 1
            },
            num_stages=3,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_M': 16,
                'BLOCK_N': 256,
                'BLOCK_K': 64,
                'SPLIT_K': 1
            },
            num_stages=3,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_M': 16,
                'BLOCK_N': 256,
                'BLOCK_K': 128,
                'SPLIT_K': 1
            },
            num_stages=3,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_M': 128,
                'BLOCK_N': 256,
                'BLOCK_K': 64,
                'SPLIT_K': 1
            },
            num_stages=3,
            num_warps=8),
        triton.Config(
            {
                'BLOCK_M': 64,
                'BLOCK_N': 256,
                'BLOCK_K': 32,
                'SPLIT_K': 1
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_M': 128,
                'BLOCK_N': 128,
                'BLOCK_K': 32,
                'SPLIT_K': 1
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_M': 128,
                'BLOCK_N': 64,
                'BLOCK_K': 32,
                'SPLIT_K': 1
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_M': 64,
                'BLOCK_N': 128,
                'BLOCK_K': 32,
                'SPLIT_K': 1
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_M': 128,
                'BLOCK_N': 32,
                'BLOCK_K': 32,
                'SPLIT_K': 1
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_M': 64,
                'BLOCK_N': 32,
                'BLOCK_K': 32,
                'SPLIT_K': 1
            },
            num_stages=5,
            num_warps=2),
        triton.Config(
            {
                'BLOCK_M': 32,
                'BLOCK_N': 64,
                'BLOCK_K': 32,
                'SPLIT_K': 1
            },
            num_stages=5,
            num_warps=2),
    ] + get_configs_io_bound3(),
    key=['M', 'N', 'K'],
    prune_configs_by={
        'early_config_prune': early_config_prune,
        'perf_model': estimate_matmul_time,
        'top_k': 10
    },
)
@triton.jit
def _linear(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    rms_scale_ptr,
    linear_scale_ptr,
):

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c = accumulator.to(tl.float32)

    rms_scale = tl.load(rms_scale_ptr + offs_am)[:, None]
    # linear_scale = tl.load(linear_scale_ptr)
    linear_scale = tl.load(linear_scale_ptr + offs_bn)[None, :]
    c = c * rms_scale * linear_scale

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.autotune(
    configs=[
        # basic configs for compute-bound matmuls
        Config({
            'BLOCK_M': 128,
            'BLOCK_N': 256,
            'BLOCK_K': 32,
            'SPLIT_K': 1
        },
               num_stages=3,
               num_warps=8),
        Config({
            'BLOCK_M': 256,
            'BLOCK_N': 128,
            'BLOCK_K': 32,
            'SPLIT_K': 1
        },
               num_stages=3,
               num_warps=8),
        Config({
            'BLOCK_M': 256,
            'BLOCK_N': 64,
            'BLOCK_K': 32,
            'SPLIT_K': 1
        },
               num_stages=4,
               num_warps=4),
        Config({
            'BLOCK_M': 64,
            'BLOCK_N': 256,
            'BLOCK_K': 32,
            'SPLIT_K': 1
        },
               num_stages=4,
               num_warps=4),
        Config({
            'BLOCK_M': 128,
            'BLOCK_N': 128,
            'BLOCK_K': 32,
            'SPLIT_K': 1
        },
               num_stages=4,
               num_warps=4),
        Config({
            'BLOCK_M': 128,
            'BLOCK_N': 64,
            'BLOCK_K': 32,
            'SPLIT_K': 1
        },
               num_stages=4,
               num_warps=4),
        Config({
            'BLOCK_M': 64,
            'BLOCK_N': 128,
            'BLOCK_K': 32,
            'SPLIT_K': 1
        },
               num_stages=4,
               num_warps=4),
        Config({
            'BLOCK_M': 128,
            'BLOCK_N': 32,
            'BLOCK_K': 32,
            'SPLIT_K': 1
        },
               num_stages=4,
               num_warps=4),
        Config({
            'BLOCK_M': 64,
            'BLOCK_N': 32,
            'BLOCK_K': 32,
            'SPLIT_K': 1
        },
               num_stages=5,
               num_warps=2),
        # good for int8
        Config({
            'BLOCK_M': 128,
            'BLOCK_N': 256,
            'BLOCK_K': 128,
            'SPLIT_K': 1
        },
               num_stages=3,
               num_warps=8),
        Config({
            'BLOCK_M': 256,
            'BLOCK_N': 128,
            'BLOCK_K': 128,
            'SPLIT_K': 1
        },
               num_stages=3,
               num_warps=8),
        Config({
            'BLOCK_M': 256,
            'BLOCK_N': 64,
            'BLOCK_K': 128,
            'SPLIT_K': 1
        },
               num_stages=4,
               num_warps=4),
        Config({
            'BLOCK_M': 64,
            'BLOCK_N': 256,
            'BLOCK_K': 128,
            'SPLIT_K': 1
        },
               num_stages=4,
               num_warps=4),
        Config({
            'BLOCK_M': 128,
            'BLOCK_N': 128,
            'BLOCK_K': 128,
            'SPLIT_K': 1
        },
               num_stages=4,
               num_warps=4),
        Config({
            'BLOCK_M': 128,
            'BLOCK_N': 64,
            'BLOCK_K': 64,
            'SPLIT_K': 1
        },
               num_stages=4,
               num_warps=4),
        Config({
            'BLOCK_M': 64,
            'BLOCK_N': 128,
            'BLOCK_K': 64,
            'SPLIT_K': 1
        },
               num_stages=4,
               num_warps=4),
        Config({
            'BLOCK_M': 128,
            'BLOCK_N': 32,
            'BLOCK_K': 64,
            'SPLIT_K': 1
        },
               num_stages=4,
               num_warps=4),
        Config({
            'BLOCK_M': 64,
            'BLOCK_N': 32,
            'BLOCK_K': 64,
            'SPLIT_K': 1
        },
               num_stages=5,
               num_warps=2),
        triton.Config(
            {
                'BLOCK_M': 32,
                'BLOCK_N': 64,
                'BLOCK_K': 128,
                'SPLIT_K': 1
            },
            num_stages=3,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_M': 16,
                'BLOCK_N': 32,
                'BLOCK_K': 128,
                'SPLIT_K': 1
            },
            num_stages=3,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_M': 16,
                'BLOCK_N': 32,
                'BLOCK_K': 64,
                'SPLIT_K': 1
            },
            num_stages=3,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_M': 16,
                'BLOCK_N': 64,
                'BLOCK_K': 128,
                'SPLIT_K': 1
            },
            num_stages=3,
            num_warps=8),
        triton.Config(
            {
                'BLOCK_M': 16,
                'BLOCK_N': 64,
                'BLOCK_K': 128,
                'SPLIT_K': 1
            },
            num_stages=3,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_M': 16,
                'BLOCK_N': 64,
                'BLOCK_K': 128,
                'SPLIT_K': 1
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_M': 16,
                'BLOCK_N': 128,
                'BLOCK_K': 128,
                'SPLIT_K': 1
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_M': 16,
                'BLOCK_N': 128,
                'BLOCK_K': 256,
                'SPLIT_K': 1
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_M': 32,
                'BLOCK_N': 128,
                'BLOCK_K': 128,
                'SPLIT_K': 1
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_M': 64,
                'BLOCK_N': 128,
                'BLOCK_K': 128,
                'SPLIT_K': 1
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_M': 32,
                'BLOCK_N': 64,
                'BLOCK_K': 128,
                'SPLIT_K': 1
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_M': 64,
                'BLOCK_N': 64,
                'BLOCK_K': 128,
                'SPLIT_K': 1
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_M': 16,
                'BLOCK_N': 128,
                'BLOCK_K': 64,
                'SPLIT_K': 1
            },
            num_stages=3,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_M': 16,
                'BLOCK_N': 256,
                'BLOCK_K': 64,
                'SPLIT_K': 1
            },
            num_stages=3,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_M': 16,
                'BLOCK_N': 256,
                'BLOCK_K': 128,
                'SPLIT_K': 1
            },
            num_stages=3,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_M': 128,
                'BLOCK_N': 256,
                'BLOCK_K': 64,
                'SPLIT_K': 1
            },
            num_stages=3,
            num_warps=8),
        triton.Config(
            {
                'BLOCK_M': 64,
                'BLOCK_N': 256,
                'BLOCK_K': 32,
                'SPLIT_K': 1
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_M': 128,
                'BLOCK_N': 128,
                'BLOCK_K': 32,
                'SPLIT_K': 1
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_M': 128,
                'BLOCK_N': 64,
                'BLOCK_K': 32,
                'SPLIT_K': 1
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_M': 64,
                'BLOCK_N': 128,
                'BLOCK_K': 32,
                'SPLIT_K': 1
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_M': 128,
                'BLOCK_N': 32,
                'BLOCK_K': 32,
                'SPLIT_K': 1
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_M': 64,
                'BLOCK_N': 32,
                'BLOCK_K': 32,
                'SPLIT_K': 1
            },
            num_stages=5,
            num_warps=2),
        triton.Config(
            {
                'BLOCK_M': 32,
                'BLOCK_N': 64,
                'BLOCK_K': 32,
                'SPLIT_K': 1
            },
            num_stages=5,
            num_warps=2),
    ] + get_configs_io_bound3(),
    key=['M', 'N', 'K'],
    prune_configs_by={
        'early_config_prune': early_config_prune,
        'perf_model': estimate_matmul_time,
        'top_k': 10
    },
)
@triton.jit
def _linear_add(
    A,
    B,
    C,
    residual_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    rms_scale_ptr,
    linear_scale_ptr,
):

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c = accumulator.to(tl.float32)

    rms_scale = tl.load(rms_scale_ptr + offs_am)[:, None]
    # linear_scale = tl.load(linear_scale_ptr)
    linear_scale = tl.load(linear_scale_ptr + offs_bn)[None, :]
    c = c * rms_scale * linear_scale
    c = c.to(residual_ptr.dtype.element_ty)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    residual_ptrs = residual_ptr + stride_cm * offs_cm[:,
                                                       None] + stride_cn * offs_cn[
                                                           None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    residual = tl.load(residual_ptrs, mask=c_mask, other=0.)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c + residual, mask=c_mask)


def linear_dynamic_quant_triton_op_fast(x,
                                        b,
                                        rms_scale,
                                        linear_scale,
                                        residual=None,
                                        bias=None,
                                        output_dtype=torch.float16):
    assert x.shape[-1] == b.shape[-1]
    assert b.is_contiguous()
    a = x  # (M, K)
    b = b.t()  # (K, N)
    assert b.stride(0) == 1
    M, K = a.shape
    K, N = b.shape
    if residual is not None:
        assert residual.shape == (M, N)
        assert residual.is_contiguous()
    c = torch.empty((M, N), device=a.device, dtype=output_dtype)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(
        N, META['BLOCK_N']), )

    if residual is not None:
        _linear_add[grid](a,
                          b,
                          c,
                          residual,
                          M,
                          N,
                          K,
                          a.stride(0),
                          a.stride(1),
                          b.stride(0),
                          b.stride(1),
                          c.stride(0),
                          c.stride(1),
                          GROUP_SIZE_M=8,
                          rms_scale_ptr=rms_scale,
                          linear_scale_ptr=linear_scale)
    else:
        _linear[grid](a,
                      b,
                      c,
                      M,
                      N,
                      K,
                      a.stride(0),
                      a.stride(1),
                      b.stride(0),
                      b.stride(1),
                      c.stride(0),
                      c.stride(1),
                      GROUP_SIZE_M=8,
                      rms_scale_ptr=rms_scale,
                      linear_scale_ptr=linear_scale)
    if bias is not None:
        c += bias

    return c


@triton.jit
def silu(x):
    return x * tl.sigmoid(x)


@triton.autotune(configs=get_configs_matmul(), key=['M', 'N', 'K'])
@triton.heuristics({
    'EVEN_K':
    lambda args: args['K'] % (args['BLOCK_K'] * args['SPLIT_K']) == 0,
})
@triton.jit
def _swiglu_int8(x_q_ptr, w1_q_ptr, w2_q_ptr, x_s_ptr, w1_s_ptr, w2_s_ptr,
                 y_ptr, M, N, K, stride_xm, stride_xk, stride_wk, stride_wn,
                 stride_ym, stride_yn, BLOCK_M: tl.constexpr,
                 BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                 GROUP_M: tl.constexpr, SPLIT_K: tl.constexpr,
                 EVEN_K: tl.constexpr):
    # matrix multiplication
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rxm = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rwn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)

    x_q_ptrs = x_q_ptr + (rxm[:, None] * stride_xm + rk[None, :] * stride_xk)
    w1_q_ptrs = w1_q_ptr + (rk[:, None] * stride_wk + rwn[None, :] * stride_wn)
    w2_q_ptrs = w2_q_ptr + (rk[:, None] * stride_wk + rwn[None, :] * stride_wn)

    acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if EVEN_K:
            x_q = tl.load(x_q_ptrs)
            w1_q = tl.load(w1_q_ptrs)
            w2_q = tl.load(w2_q_ptrs)
        else:
            k_remaining = K - k * (BLOCK_K * SPLIT_K)
            _0 = tl.zeros((1, 1), dtype=x_q_ptrs.dtype.element_ty)
            x_q = tl.load(x_q_ptrs, mask=rk[None, :] < k_remaining, other=_0)
            w1_q = tl.load(w1_q_ptrs, mask=rk[:, None] < k_remaining, other=_0)
            w2_q = tl.load(w2_q_ptrs, mask=rk[:, None] < k_remaining, other=_0)
        # We accumulate along the K dimension.
        acc1 += tl.dot(x_q, w1_q, out_dtype=tl.int32)
        acc2 += tl.dot(x_q, w2_q, out_dtype=tl.int32)
        x_q_ptrs += BLOCK_K * SPLIT_K * stride_xk
        w1_q_ptrs += BLOCK_K * SPLIT_K * stride_wk
        w2_q_ptrs += BLOCK_K * SPLIT_K * stride_wk
    x_s = tl.load(x_s_ptr + rxm, mask=rm < M, other=0.)
    w1_s = tl.load(w1_s_ptr + rwn, mask=rn < N, other=0.)
    w2_s = tl.load(w2_s_ptr + rwn, mask=rn < N, other=0.)
    y1 = silu(acc1 * x_s[:, None] * w1_s[None, :])
    y2 = acc2 * x_s[:, None] * w2_s[None, :]
    y = (y1 * y2).to(y_ptr.dtype.element_ty)

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    y_ptrs = y_ptr + (rm[:, None] * stride_ym + rn[None, :] * stride_yn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    # handles write-back with reduction-splitting
    if SPLIT_K == 1:
        tl.store(y_ptrs, y, mask=mask)
    else:
        tl.atomic_add(y_ptrs, y, mask=mask)


def swiglu_int8_tri(x_q,
                    w1_q,
                    w2_q,
                    x_s,
                    w1_s,
                    w2_s,
                    eps,
                    output_dtype=torch.float16):
    assert w1_q.shape == w2_q.shape
    assert x_q.shape[-1] == w1_q.shape[-1]
    assert x_q.is_contiguous(), 'Matrix x_q must be contiguous'

    assert w1_q.is_contiguous(), 'Matrix w1_q.T must be contiguous'
    w1_q = w1_q.T
    assert w1_q.stride(0) == 1
    assert w2_q.is_contiguous(), 'Matrix w2_q.T must be contiguous'
    w2_q = w2_q.T
    assert w2_q.stride(0) == 1

    M, K = x_q.shape
    N = w1_q.shape[1]
    y = torch.empty((M, N), device=x_q.device, dtype=output_dtype)

    # launch kernel
    def grid(META):
        return (triton.cdiv(M, META['BLOCK_M']) *
                triton.cdiv(N, META['BLOCK_N']), META['SPLIT_K'])

    _swiglu_int8[grid](x_q,
                       w1_q,
                       w2_q,
                       x_s,
                       w1_s,
                       w2_s,
                       y,
                       M,
                       N,
                       K,
                       x_q.stride(0),
                       x_q.stride(1),
                       w1_q.stride(0),
                       w1_q.stride(1),
                       y.stride(0),
                       y.stride(1),
                       GROUP_M=8)

    return y


@triton.jit
def _per_token_quant_int8(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    y_stride,
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    y_ptr += row * y_stride
    y_q_ptr += row * y_stride
    y_s_ptr += row

    cols = tl.arange(0, BLOCK)  # N <= BLOCK
    mask = cols < N

    y = tl.load(y_ptr + cols, mask=mask, other=0.).to(tl.float32)
    # Quant
    # _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    _absmax = tl.max(tl.abs(y))
    y_s = _absmax / 127
    y_q = tl.maximum(tl.minimum(tl.math.round(y / y_s), 127), -128).to(tl.int8)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


def per_token_quant_int8_tri(x, eps):
    M, N = x.shape
    x_q = torch.empty((M, N), device=x.device, dtype=torch.int8)
    x_q_arg = x_q.reshape(-1, x_q.shape[-1])
    x_s = torch.empty(M, device=x.device, dtype=torch.float32)
    BLOCK = triton.next_power_of_2(N)
    # heuristics for number of warps
    num_warps = min(max(BLOCK // 256, 1), 8)
    # enqueue kernel
    _per_token_quant_int8[(M, )](x,
                                 x_q,
                                 x_s,
                                 x_q_arg.stride(0),
                                 N,
                                 eps,
                                 BLOCK=BLOCK,
                                 num_warps=num_warps)
    return x_q, x_s


@triton.jit
def _rms_norm_fwd_fused_dynamic_symmetric(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    Scale,
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    cols = tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
    _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    w = tl.load(W + cols, mask=mask)
    x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
    x_hat = x * rstd
    y = x_hat * w

    # scale = tl.maximum(tl.max(tl.abs(y)), eps).to(tl.float32) / 127
    scale = tl.max(tl.abs(y)).to(tl.float32) / 127
    tl.store(Scale + row, scale)

    y = tl.math.round(y / scale)
    y = tl.minimum(y, 127)
    y = tl.maximum(y, -128)
    tl.store(Y + cols, y, mask=mask)


def rms_norm_dynamic_quant(x, w, eps):
    x_arg = x.reshape(-1, x.shape[-1])
    # y = torch.empty_like(x, dtype=torch.float16)
    y = torch.empty_like(x_arg, dtype=torch.int8)
    M, K = x_arg.shape
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(K))
    if K > BLOCK_SIZE:
        raise RuntimeError(
            "This rms norm doesn't support feature dim >= 64KB.")
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    scale = torch.empty((M, 1), dtype=torch.float32, device=x.device)
    _rms_norm_fwd_fused_dynamic_symmetric[(M, )](
        x_arg,
        y,
        w,
        scale,
        x_arg.stride(0),
        K,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return y, scale


def mlp_int8_tri(x1,
                 w_norm,
                 w1_q,
                 w2_q,
                 w3_q,
                 w1_s,
                 w2_s,
                 w3_s,
                 eps,
                 output_dtype=torch.float16):
    x1_args = x1.view(-1, x1.shape[-1])
    M, N = x1_args.shape
    K = w1_q.shape[0]
    y2_q, y2_s = rms_norm_dynamic_quant(x1_args, w_norm, eps)
    if (M <= 32 and ((N <= 8192 and K <= 13824) or
                     (N <= 5120 and K <= 14336) or (N <= 4096 and K <= 28672))
            or M <= 64 and ((N <= 8192 and K <= 11008) or
                            (N <= 5120 and K <= 14336) or
                            (N <= 4096 and K <= 14336)) or M <= 128 and
        ((N <= 8192 and K <= 7168) or (N <= 5120 and K <= 13824) or
         (N <= 4096 and K <= 14336)) or M <= 256 and
        ((N <= 8192 and K <= 5504) or (N <= 5120 and K <= 7168) or
         (N <= 4096 and K <= 7168)) or M <= 512 and
        ((N <= 8192 and K <= 2752) or (N <= 5120 and K <= 3584) or
         (N <= 4096 and K <= 3584))):
        y3 = swiglu_int8_tri(y2_q, w1_q, w3_q, y2_s, w1_s, w3_s, eps,
                             output_dtype)
    else:
        y2_1 = linear_dynamic_quant_triton_op_fast(y2_q,
                                                   w1_q,
                                                   y2_s,
                                                   w1_s,
                                                   output_dtype=output_dtype)
        y2_2 = linear_dynamic_quant_triton_op_fast(y2_q,
                                                   w3_q,
                                                   y2_s,
                                                   w3_s,
                                                   output_dtype=output_dtype)
        y3 = silu_elem_mul_tri(y2_1, y2_2)
    y3_q, y3_s = per_token_quant_int8_tri(y3, eps)
    out = linear_dynamic_quant_triton_op_fast(y3_q,
                                              w2_q,
                                              y3_s,
                                              w2_s,
                                              residual=x1_args,
                                              output_dtype=output_dtype)
    out = out.view_as(x1)
    return out


def add_rmsnorm_pt(x1, x2, weight, eps):
    add_x1x2 = x1  #+ x2
    norm_x = add_x1x2.float() * torch.rsqrt(
        add_x1x2.float().pow(2).mean(-1, keepdim=True) + eps)
    output = norm_x.type_as(add_x1x2)
    return add_x1x2, output * weight


def swiglu_pt(x, w1, w2):
    return F.silu(x @ w1) * (x @ w2)


def linear_add_pt(x1, w, x2):
    return x1 @ w + x2


def mlp_pt(x1, x2, w_norm, w1, w2, w3, eps):
    y1, y2 = add_rmsnorm_pt(x1, x2, w_norm, eps)
    y2 = swiglu_pt(y2, w1.T, w3.T)
    out = linear_add_pt(y2, w2.T, y1)
    return out


def test_mlp(B, M, N, K, eps=1e-5):
    x1 = torch.randn((B, M, N), device='cuda', dtype=torch.float16) * 0.2
    x2 = torch.randn((B, M, N), device='cuda', dtype=torch.float16) * 0.2
    w_norm = torch.randn((N, ), device='cuda', dtype=torch.float16) * 0.05
    w1 = torch.randn((K, N), device='cuda', dtype=torch.float16) * 0.05
    w2 = torch.randn((N, K), device='cuda', dtype=torch.float16) * 0.05
    w3 = torch.randn((K, N), device='cuda', dtype=torch.float16) * 0.05

    w1_q, w1_s = per_channel_quant(w1, 8, eps, torch.int8)
    print(w1_q.shape, w1_s.shape)
    w2_q, w2_s = per_channel_quant(w2, 8, eps, torch.int8)
    w3_q, w3_s = per_channel_quant(w3, 8, eps, torch.int8)

    y_tri_int8 = mlp_int8_tri(
        x1,
        # x2,
        w_norm,
        w1_q,
        w2_q,
        w3_q,
        w1_s,
        w2_s,
        w3_s,
        eps)
    y_ref_fp16 = mlp_pt(x1, x2, w_norm, w1, w2, w3, eps)
    print(y_tri_int8)
    print(y_ref_fp16)
    assert torch.allclose(y_tri_int8, y_ref_fp16, atol=1e-1, rtol=1e-1)


def quantize_int8(weight, axis=0):
    # Weight shape: [H1, H2]
    # Scale shape: [H2]
    scale = weight.abs().amax(axis, keepdim=True) / 127.
    weight = (weight / scale).to(torch.int8)
    # col major will accelerate i8xi8 kernel.
    if axis == 0:
        weight = weight.t().contiguous().t()
    scale = scale.squeeze(axis)
    return weight, scale, None


def test_rms_and_linear(M, N, K, dtype=torch.float16, eps=1e-5, device='cuda'):

    def rms_norm_torch(x, w, eps):
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        # print(w * x)
        return w * x

    def linear_torch(x, b):
        return F.linear(x, b)

    torch.manual_seed(0)
    x_shape = (M, K)
    rms_w_shape = (x_shape[-1], )
    # rms_weight = torch.randn(rms_w_shape, dtype=dtype, device='cuda', requires_grad=True)
    rms_weight = torch.load('rms_weight.pth').cuda()
    x = torch.load('hidden_states.pth').cuda()
    linear_weight = torch.load('v_proj_weight.pth').cuda()
    # x = torch.randn(x_shape, dtype=dtype, device='cuda')
    # linear_weight = torch.randn((N, K), dtype=dtype, device='cuda', requires_grad=True)
    # linear_weight_quant, linear_scale, _ = quantize_int8(linear_weight,1)
    linear_weight_quant, linear_scale = per_channel_quant(
        linear_weight, 8, 1e-5, torch.int8)
    # linear_alpha = linear_weight.abs().max(dim=1)[0]
    # linear_scale = linear_alpha / 127
    # linear_weight_quant = torch.clamp(torch.round(linear_weight / linear_scale[:, None]), -127, 127).to(torch.int8)
    # import pdb;pdb.set_trace()

    x = x.view(-1, x.shape[-1])
    rms_out, rms_scale = rms_norm_dynamic_quant(x, rms_weight, eps)
    linear_out = linear_dynamic_quant_triton_op_fast(rms_out,
                                                     linear_weight_quant,
                                                     rms_scale,
                                                     linear_scale,
                                                     output_dtype=dtype)
    rms_out_torch = rms_norm_torch(x, rms_weight, eps).half()
    linear_out_torch = linear_torch(rms_out_torch, linear_weight)
    print(f'linear_out.abs().mean() = {linear_out.abs().mean()}')
    print(f'linear_out_torch.abs().mean() = {linear_out_torch.abs().mean()}')
    print('perchannel error: ', (linear_out - linear_out_torch).abs().mean())
    cos = torch.nn.CosineSimilarity(0)
    print(
        'Output cos',
        cos(linear_out.flatten().to(torch.float32),
            linear_out_torch.flatten().to(torch.float32)))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M'],
        # x_vals=[2 ** i for i in range(6, 16)],
        x_vals=[1, 16, 32, 64, 128, 256],
        #     + [
        #     512 * i * 2 for i in range(1, 17)
        # ],
        line_arg='provider',
        line_vals=['int8_dynamic_triton_op', 'float_torch'],
        line_names=['int8_dynamic_triton_op', 'float_torch'],
        styles=[('blue', '-'), ('green', '-'), ('orange', '-'),
                ('yellow', '-'), ('yellow', '-')],
        ylabel='GB/s',
        plot_name='forward',
        args={
            'dtype': torch.float16,
        }))
def bench_rms_and_linear(M, dtype, provider, eps=1e-5, device='cuda'):

    def rms_norm_torch(x, w, eps):
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        # print(w * x)
        return w * x

    def linear_torch(x, b):
        return F.linear(x, b)

    N = 4096
    K = 4096
    x_shape = (M, K)
    rms_w_shape = (x_shape[-1], )
    rms_weight = torch.randn(rms_w_shape,
                             dtype=dtype,
                             device='cuda',
                             requires_grad=True)
    x = torch.randn(x_shape, dtype=dtype, device='cuda')
    linear_weight = torch.randn((N, K),
                                dtype=dtype,
                                device='cuda',
                                requires_grad=True)
    linear_weight_quant, linear_scale = per_channel_quant(
        linear_weight, 8, 1e-5, torch.int8)

    alpha = max(x.max().abs(), x.min().abs())
    rms_scale = alpha / 127

    if provider == 'int8_dynamic_triton_op':
        rms_out, rms_scale = rms_norm_dynamic_quant(x, rms_weight, eps)

        def y_fwd():

            linear_out = linear_dynamic_quant_triton_op_fast(
                rms_out,
                linear_weight_quant,
                rms_scale,
                linear_scale,
                output_dtype=dtype)
    elif provider == 'float_torch':
        rms_out_torch = rms_norm_torch(x, rms_weight, eps).half()

        def y_fwd():

            linear_out_torch = linear_torch(rms_out_torch, linear_weight)

    tflops = lambda ms: (x.element_size() * M * N * K * 1e-12) / (ms * 1e-3)
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(y_fwd,
                                                 quantiles=quantiles,
                                                 rep=500)
    # return tflops(ms), tflops(max_ms), tflops(min_ms)
    return ms, max_ms, min_ms


def rms_norm_torch(x, w, eps):
    variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    # print(w * x)
    return w * x


def linear_torch(x, b):
    return F.linear(x, b)


import time


def bench_rms_and_linear1(M, eps=1e-5, device='cuda'):
    N = 4096
    K = 4096
    dtype = torch.float16
    x_shape = (M, K)
    rms_w_shape = (x_shape[-1], )
    rms_weight = torch.randn(rms_w_shape,
                             dtype=dtype,
                             device='cuda',
                             requires_grad=True)
    x = torch.randn(x_shape, dtype=dtype, device='cuda')
    linear_weight = torch.randn((N, K),
                                dtype=dtype,
                                device='cuda',
                                requires_grad=True)
    linear_weight_quant, linear_scale = per_channel_quant(
        linear_weight, 8, 1e-5, torch.int8)

    alpha = max(x.max().abs(), x.min().abs())
    rms_scale = alpha / 127

    for _ in range(10):
        rms_out, rms_scale = rms_norm_dynamic_quant(x, rms_weight, eps)
        linear_out = linear_dynamic_quant_triton_op_fast(
            rms_out,
            linear_weight_quant,
            rms_scale,
            linear_scale,
            output_dtype=torch.float16)
    torch.cuda.synchronize()

    iters = 512
    t1 = time.time()
    for _ in range(iters):
        rms_out, rms_scale = rms_norm_dynamic_quant(x, rms_weight, eps)
        linear_out = linear_dynamic_quant_triton_op_fast(
            rms_out,
            linear_weight_quant,
            rms_scale,
            linear_scale,
            output_dtype=torch.float16)
    t2 = time.time()
    print(f'triton cost time {t2 - t1}')

    for _ in range(10):
        rms_out_torch = rms_norm_torch(x, rms_weight, eps).half()
        linear_out_torch = linear_torch(rms_out_torch, linear_weight)
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(iters):
        rms_out_torch = rms_norm_torch(x, rms_weight, eps).half()
        linear_out_torch = linear_torch(rms_out_torch, linear_weight)
    torch.cuda.synchronize()
    t2 = time.time()
    print(f'torch cost time {t2 - t1}')


if __name__ == '__main__':
    torch.manual_seed(0)
    # test_mlp(1, 129, 4096, 11008)
    test_rms_and_linear(4, 2048, 4096)
    # bench_rms_and_linear.run(print_data=True)
    # for m in [1, 16, 32, 64, 128, 256]:
    # bench_rms_and_linear.run(print_data=True)
