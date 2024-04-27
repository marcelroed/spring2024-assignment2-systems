import torch
import triton
import triton.language as tl
import torch.nn as nn


def compute_rmsnorm_backward_g(grad_out, x, g):
    xrms = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)
    dims = tuple(range(0, len(x.shape) - 1))
    return (xrms * grad_out).sum(dims)


def compute_rmsnorm_backward_x(grad_out, x, g):
    x_shape = x.shape
    d = x.shape[-1]
    x = x.view(-1, d)
    grad_out = grad_out.view(-1, d)

    gj = g[None, :]
    ms = x.pow(2).mean(-1, keepdim=True) + 1e-5

    gxgrad = (x * gj * grad_out).sum(-1, keepdim=True)

    out = (gj * grad_out - x * gxgrad / (d * ms)) * torch.rsqrt(ms)
    print(out.shape)
    return out.view(*x_shape)

class RMSNormAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, g):
        ctx.save_for_backward(x, g)
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)
        x = x * rms
        return g * x
    
    @staticmethod
    def backward(ctx, grad_out):
        x, g = ctx.saved_tensors
        grad_g = compute_rmsnorm_backward_g(grad_out, x, g)
        grad_x = compute_rmsnorm_backward_x(grad_out, x, g)
        return grad_x, grad_g


@triton.jit
def rmsnorm_forward(
    x_ptr: tl.pointer_type, # [N, H]
    g_ptr: tl.pointer_type,  # [H]
    out_ptr: tl.pointer_type,  # [N, H]
    x_row_stride: tl.uint32,
    eps: tl.float32,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start_ptr = x_ptr + row_idx * x_row_stride
    offsets = tl.arange(0, BLOCK_SIZE)
    x_ptrs = row_start_ptr + offsets
    mask = offsets < x_row_stride

    x_vals = tl.load(x_ptrs, mask=mask, other=0.0)
    x_sqr = x_vals * x_vals
    x_sum = tl.sum(x_sqr, axis=0)
    x_mean = x_sum / x_row_stride + eps

    out_start_ptr = out_ptr + row_idx * x_row_stride
    out_ptrs = out_start_ptr + offsets

    normalized = x_vals / tl.sqrt(x_mean)
    gate = tl.load(g_ptr + offsets, mask=mask, other=0.0)
    gated = normalized * gate

    tl.store(out_ptrs, gated, mask=mask)

@triton.jit
def rmsnorm_backward(
    grad_out_ptr: tl.pointer_type,  # [N, H]
    x_ptr: tl.pointer_type,  # [N, H]
    g_ptr: tl.pointer_type,  # [H]
    grad_x_ptr: tl.pointer_type,  # [N, H]
    grad_g_partial_ptr: tl.pointer_type,  # [GROUP_SIZE_M, H]
    Lock: tl.pointer_type,
    x_row_stride: tl.uint32,
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    row_idx = tl.program_id(0)

    lock_id = row_idx % GROUP_SIZE_M
    Lock += lock_id

    row_start_ptr = x_ptr + row_idx * x_row_stride
    offsets = tl.arange(0, BLOCK_SIZE_N)
    x_ptrs = row_start_ptr + offsets
    mask = offsets < x_row_stride

    x_vals = tl.load(x_ptrs, mask=mask, other=0.0)
    x_sqr = x_vals * x_vals
    x_sum = tl.sum(x_sqr, axis=0)
    ms = x_sum / x_row_stride + 1e-5

    rms = tl.sqrt(ms)
    xrms = x_vals / rms

    grad_out_start_ptr = grad_out_ptr + row_idx * x_row_stride
    grad_out_ptrs = grad_out_start_ptr + offsets
    grad_out = tl.load(grad_out_ptrs, mask=mask, other=0.0)

    grad_g_partial_ptrs = grad_g_partial_ptr + lock_id * x_row_stride + offsets

    grad_g = xrms * grad_out

    while tl.atomic_cas(Lock, 0, 1) == 1:  # Acquire partial result lock
        pass
    grad_g += tl.load(grad_g_partial_ptrs, mask=mask)
    tl.store(grad_g_partial_ptrs, grad_g, mask=mask)
    tl.atomic_xchg(Lock, 0)  # Release lock

    g_vals = tl.load(g_ptr + offsets, mask=mask, other=0.0)

    gxgrad = tl.sum(g_vals[None, :] * x_vals * grad_out, axis=1)

    grad_x = (g_vals * grad_out - x_vals * gxgrad / (x_row_stride * ms)) / rms

    grad_x_start_ptr = grad_x_ptr + row_idx * x_row_stride

    grad_x_ptrs = grad_x_start_ptr + offsets

    tl.store(grad_x_ptrs, grad_x, mask=mask)

@triton.jit
def rmsnorm_backward_grad_g(
    grad_g_partials_ptr: tl.pointer_type,  # [GROUP_SIZE_M, H]
    grad_g_ptr: tl.pointer_type,  # [H]
    GROUP_SIZE_M: tl.constexpr,
    x_row_stride: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    grad_g = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(0, GROUP_SIZE_M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (rows[:, None] < GROUP_SIZE_M) & (cols[None, :] < x_row_stride)
        offset = rows[:, None] * x_row_stride + cols[None, :]
        grad_g += tl.load(grad_g_partials_ptr + offset, mask=mask, other=0.0)

    sum_grad_g = tl.sum(grad_g, axis=0)
    tl.store(grad_g_ptr + cols, sum_grad_g, mask=cols < x_row_stride)

class RMSNormAutogradFunctionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, g):
        x_shape = x.shape
        x = x.view(-1, x_shape[-1])
        N, H = x.shape
        ctx.BLOCK_SIZE = triton.next_power_of_2(H)
        out = torch.empty_like(x)
        num_warps = min(max(ctx.BLOCK_SIZE // 256, 1), 8)
        rmsnorm_forward[(N,)](x, g, out, x.stride(0), 1e-5, num_warps=num_warps, BLOCK_SIZE=ctx.BLOCK_SIZE, num_ctas=1)
        ctx.save_for_backward(x, g)
        ctx.num_warps = num_warps
        return out.view(x_shape)
    
    @staticmethod
    def backward(ctx, grad_out):
        x_flat, g = ctx.saved_tensors
        in_shape = grad_out.shape

        N, H = x_flat.shape
        grad_out = grad_out.view(-1, H)

        if H <= 8192: GROUP_SIZE_M = 96
        if H <= 4096: GROUP_SIZE_M = 128
        if H <= 1024: GROUP_SIZE_M = 256

        grad_g = torch.zeros_like(g)
        grad_g_partial = torch.zeros((GROUP_SIZE_M, H), device=g.device, dtype=g.dtype)
        lock = torch.zeros(GROUP_SIZE_M, dtype=torch.int32, device=g.device)

        grad_x = torch.empty_like(x_flat)

        rmsnorm_backward[
            (N,)
        ](
            grad_out, x_flat, g, grad_x, grad_g_partial, lock, x_row_stride=H, num_warps=ctx.num_warps, GROUP_SIZE_M=GROUP_SIZE_M, BLOCK_SIZE_N=ctx.BLOCK_SIZE,
        )

        rmsnorm_backward_grad_g[
            lambda meta: [triton.cdiv(H, meta['BLOCK_SIZE_N'])]
        ](
            grad_g_partial, grad_g, min(GROUP_SIZE_M, N), H, BLOCK_SIZE_M=32, BLOCK_SIZE_N=128, num_ctas=1
        )

        grad_x = grad_x.view(in_shape)

        return grad_x, grad_g


class RMSNormTriton(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
    
    def forward(self, x):
        return RMSNormAutogradFunctionTriton.apply(x, self.weight)
