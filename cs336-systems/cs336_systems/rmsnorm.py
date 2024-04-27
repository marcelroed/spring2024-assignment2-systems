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
    i = torch.arange(d, device=x.device)[None, None, :]
    j = torch.arange(d, device=x.device)[None, :, None]
    xi = x[:, None, :]
    xj = x[:, :, None]
    gi = g[None, None, :]
    ms = (x.pow(2).mean(-1, keepdim=True) + 1e-5)[..., None]
    out = (((i == j).float() - xi * xj / (ms * d)) * torch.rsqrt(ms) * gi * grad_out[..., None, :]).sum(-1)
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
    grad_g_ptr: tl.pointer_type,  # [H]
    x_row_stride: tl.uint32,
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
    x_mean = x_sum / x_row_stride + 1e-5
    rms = tl.sqrt(x_mean)

    xrms = x_vals / rms
    tl.sto

    grad_out_start_ptr = grad_out_ptr + row_idx * x_row_stride

    grad_out_ptrs = grad_out_start_ptr + offsets

    grad_out_vals = tl.load(grad_out_ptrs, mask=mask, other=0.0)

    grad_g = tl.sum(grad_out_vals * x_vals, axis=0)

    tl.store(grad_g_ptr + offsets, grad_g, mask=mask)

    grad_x_start_ptr = grad_x_ptr + row_idx * x_row_stride

    grad_x_ptrs = grad_x_start_ptr + offsets

    normalized = x_vals / tl.sqrt(x_mean)

    gate = tl.load(g_ptr + offsets, mask=mask, other=0.0)

    grad_x = (grad_out_vals * gate) * (1.0 - x_vals * x_vals / (x_mean * x_row_stride))

    tl.store(grad_x_ptrs, grad_x, mask=mask)
    

class RMSNormAutogradFunctionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, g):
        x_shape = x.shape
        x = x.view(-1, x_shape[-1])
        N, H = x.shape
        ctx.save_for_backward(x, g)
        ctx.BLOCK_SIZE = triton.next_power_of_2(H)
        out = torch.empty_like(x)
        rmsnorm_forward[(N,)](x, g, out, x.stride(0), 1e-5, num_warps=16, BLOCK_SIZE=ctx.BLOCK_SIZE)
        return out.view(x_shape)
    
    @staticmethod
    def backward(ctx, grad_out):
        x, g = ctx.saved_tensors
        x_shape = x.shape
        d = x.shape[-1]
        x = x.view(-1, d)
        N = x.shape[0]
        grad_g = torch.empty(d, device=x.device)
        grad_x = torch.empty_like(x)
        rmsnorm_backward[(N,)](grad_out, x, g, grad_x, grad_g, x_row_stride=x_shape, num_warps=16, BLOCK_SIZE=ctx.BLOCK_SIZE)
        return grad_x.view(x_shape), grad_g


class RMSNormTriton(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
    
    def forward(self, x):
        return RMSNormAutogradFunctionTriton.apply(x, self.weight)
