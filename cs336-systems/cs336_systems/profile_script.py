import torch
import torch.nn as nn
from argparse import ArgumentParser
from typing import Literal
import numpy as np
from cs336_basics.model import BasicsTransformerLM, RMSNorm
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.rmsnorm import RMSNormTriton
from timeit import default_timer as timer
from contextlib import nullcontext
from torch.profiler import profile, record_function, ProfilerActivity
import gc

def format_bytes(size):
    power = 2**10
    n = 0
    power_labels = {0: 'B', 1: 'KiB', 2: 'MiB', 3: 'GiB'}
    while size > power:
        size /= power
        n += 1
    return f'${size:.2f}$ {power_labels[n]}'


def initialize_model(
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int | None = None,
    vocab_size: int = 10_000,
    context_length: int = 128,
    attn_pdrop: float | None = None,
    residual_pdrop: float | None = None,
    norm_class: type = RMSNorm,
    device='cuda',
) -> BasicsTransformerLM:
    if d_ff is None:
        d_ff = 4 * d_model
    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        attn_pdrop=attn_pdrop,
        residual_pdrop=residual_pdrop,
        norm_class=norm_class,
    )
    return model.to(device)

def get_random_batch(batch_size: int, context_length: int, vocab_size: int, device='cuda') -> tuple[torch.Tensor, torch.Tensor]:
    return torch.randint(0, vocab_size, (batch_size, context_length), device=device), torch.randint(0, vocab_size, (batch_size, context_length), device=device)

def profile_model(model, batch, warmup_steps, profile_steps, passes: Literal['forward', 'backward', 'both'], mixed_precision=False):
    mp = torch.autocast('cuda', dtype=torch.bfloat16) if mixed_precision else nullcontext()
    with mp:
        for _ in range(warmup_steps):
            loss = cross_entropy(model(batch[0]), batch[1])
            if passes == 'backward' or passes == 'both':
                loss.backward()
            torch.cuda.synchronize()
        measurement_results = np.zeros(profile_steps)
        for i in range(profile_steps):
            if passes == 'forward':
                start = timer()
                loss = cross_entropy(model(batch[0]), batch[1])
                # out = model(batch[0])
            elif passes == 'backward':
                loss = cross_entropy(model(batch[0]), batch[1])
                torch.cuda.synchronize()
                start = timer()
                loss.backward()
            elif passes == 'both':
                start = timer()
                loss = cross_entropy(model(batch[0]), batch[1])
                loss.backward()
            torch.cuda.synchronize()
            end = timer()
            time = end - start
            measurement_results[i] = time
        mean_time = np.mean(measurement_results)
        std = np.std(measurement_results)
        # print(f'{passes} mean time: {mean_time:.6f} s, std: {std:.6f} s')
    return mean_time, std

configs = {
    'small': {'d_model': 768, 'num_layers': 12, 'num_heads': 12},
    'medium': {'d_model': 1024, 'num_layers': 24, 'num_heads': 16},
    'large': {'d_model': 1280, 'num_layers': 36, 'num_heads': 20},
    'xl': {'d_model': 1600, 'num_layers': 48, 'num_heads': 25},
    '2.7B': {'d_model': 2560, 'num_layers': 32, 'num_heads': 32},
}

def perform_all_profiles(include_warmup: bool, mixed_precision=False, norm_class=RMSNorm):
    batch = get_random_batch(batch_size=args.batch_size, context_length=128, vocab_size=10_000)
    for key, value in configs.items():
        model = initialize_model(**value, norm_class=norm_class)  # Reinitialize every loop to show effect of warmup
        for p in ['forward', 'backward', 'both']:
            mean, std = profile_model(model, batch, warmup_steps=1 if include_warmup else 0, profile_steps=5, passes=p, mixed_precision=mixed_precision)
            print(f'[{key}], [{p}], [${mean:.2e}$], [${std:.2e}$],')
            gc.collect()
            torch.cuda.empty_cache()
        del model
        gc.collect()
        torch.cuda.empty_cache()

def perform_all_profiles_norm(include_warmup: bool, passes: list[str] = ['forward', 'backward'], mixed_precision=False, compile=False):
    batch = get_random_batch(batch_size=args.batch_size, context_length=128, vocab_size=10_000)
    for key, value in configs.items():
        for norm_class in [RMSNorm, nn.LayerNorm, RMSNormTriton] if not compile else [RMSNorm]:
            model = initialize_model(**value, norm_class=norm_class)  # Reinitialize every loop to show effect of warmup
            if compile:
                model = torch.compile(model)
            # for p in ['forward', 'backward', 'both']:
            for p in passes:
                mean, std = profile_model(model, batch, warmup_steps=5 if include_warmup else 0, profile_steps=5, passes=p, mixed_precision=mixed_precision)
                print(f'[{key}], [{p}], [{norm_class.__name__}], [${mean:.2e}$], [${std:.2e}$],')
                gc.collect()
                torch.cuda.empty_cache()
            del model
            gc.collect()
            torch.cuda.empty_cache()

def run_step(model, batch, optimizer, enable_backward: bool, enable_optimizer: bool, mixed_precision=False):
    context = torch.autocast('cuda', dtype=torch.bfloat16) if mixed_precision else nullcontext()
    with context:
        with record_function('forward_pass'):
            loss = cross_entropy(model(batch[0]), batch[1])
        
        if enable_backward:
            with record_function('backward_pass'):
                loss.backward()
            if enable_optimizer:
                with record_function('optimizer'):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

def run_pytorch_profile(enable_backward=True, enable_optimizer=True):
    model = initialize_model(**configs['xl'])
    batch = get_random_batch(batch_size=args.batch_size, context_length=128, vocab_size=10_000)
    optimizer = AdamW(model.parameters(), lr=1e-4)

    # warmup
    loss = cross_entropy(model(batch[0]), batch[1])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    n_steps = 5
    with profile(activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ], experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
    record_shapes=True,
    profile_memory=False,
    with_stack=True,
    ) as prof:
        for _ in range(n_steps):
            run_step(model, batch, optimizer, enable_backward=enable_backward, enable_optimizer=enable_optimizer)
            prof.step()
    prof.export_stacks('lm_profiler_stacks.txt', 'self_cuda_time_total')
    print(prof.key_averages().table(max_name_column_width=120, sort_by='cpu_time_total', row_limit=50))
    print(prof.key_averages().table(max_name_column_width=300, sort_by='cuda_time_total', row_limit=50))

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        print(f"{self.fc1.weight.dtype=}")
        print(x.dtype)
        x = self.fc1(x)
        print('Post fc1', x.dtype)
        x = self.relu(x)
        print('Post relu', x.dtype)
        x = self.ln(x)
        print('Post ln', x.dtype)
        x = self.fc2(x)
        print('Post fc2', x.dtype)
        return x

def test_mp(dtype=torch.float16):
    model = ToyModel(10, 10).to('cuda')
    batch = torch.randn(16, 10, device='cuda')
    with torch.autocast('cuda', dtype=dtype):
        out = model(batch)
    
def norm_bench(include_backward: bool):
    n_rows = 50_000
    seen_rms_norm = False
    for ln_type in [nn.LayerNorm, RMSNorm, RMSNormTriton, RMSNorm]:
        for n_cols in [1024, 2048, 4096, 8192]:
            x = torch.randn(n_rows, n_cols, device='cuda')
            w = torch.randn(n_cols, device='cuda')
            dy = torch.randn_like(x, device='cuda')
            if 'RMSNorm' in ln_type.__name__:
                ln = ln_type(n_cols).to('cuda')
                ln.weight.data[:] = w
                if not seen_rms_norm and ln_type == RMSNorm:
                    ln = torch.compile(ln)
            else:
                b = torch.randn_like(w)
                ln = ln_type(n_cols).to('cuda')
                ln.weight.data[:] = w
                ln.bias.data[:] = b
            
            start = timer()
            for _ in range(5):
                y = ln(x)  # warmup
                if include_backward:
                    x.grad = None
                    y.backward(dy)
                torch.cuda.synchronize()
            for _ in range(1_000):
                y = ln(x)
                if include_backward:
                    y.grad = None
                    y.backward(dy)
                torch.cuda.synchronize()
            end = timer()
            print(f'[{ln_type.__name__ + (" Compiled" if not seen_rms_norm and RMSNorm == ln_type else "")}], [{n_cols}], [${(end - start)/1_000:.3e}$],')
        if 'RMSNorm' in ln_type.__name__:
            seen_rms_norm = True

def memory_profile(enable_backward: bool, enable_optimizer: bool, mixed_precision: bool = False):
    with (nullcontext() if enable_backward else torch.no_grad()):
        torch.cuda.memory._record_memory_history(max_entries=1_000_000)
        n_steps = 3

        model = initialize_model(**configs['2.7B'])
        device = model.lm_head.weight.device
        batch = get_random_batch(batch_size=args.batch_size, context_length=128, vocab_size=10_000)
        optimizer = AdamW(model.parameters(), lr=1e-4)

        # warmup
        loss = cross_entropy(model(batch[0]), batch[1])
        if enable_backward:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

        with profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=n_steps),
            experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for _ in range(n_steps):
                # Run on a batch
                run_step(model, batch, optimizer, enable_backward=enable_backward, enable_optimizer=enable_optimizer, mixed_precision=mixed_precision)
                prof.step()
            # Save timeline
            prof.export_memory_timeline(f'timeline{enable_backward=}{enable_optimizer=}.html', device=device)
        torch.cuda.memory._dump_snapshot(f'memory_snapshot{enable_backward=}{enable_optimizer=}.pickle')
        torch.cuda.memory._record_memory_history(enabled=None)
    
def peak_memory_usage():
    device = 'cuda:0'
    for config_name, config in configs.items():
        for full_step in [False, True]:
            with nullcontext() if full_step else torch.no_grad():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                model = initialize_model(**config, device=device)
                batch = get_random_batch(batch_size=args.batch_size, context_length=128, vocab_size=10_000, device=device)
                optimizer = AdamW(model.parameters(), lr=1e-3)
                loss = cross_entropy(model(batch[0]), batch[1])
                if full_step:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                torch.cuda.synchronize(device=device)
                peak_usage = torch.cuda.max_memory_allocated(device=device)
                print(f'[{config_name}], [{"full step" if full_step else "forward"}], [{format_bytes(peak_usage)}],')
                del model, batch, optimizer, loss

def peak_memory_usage_mixed():
    device = 'cuda:0'
    config = configs['2.7B']
    for full_step in [False, True]:
        with (nullcontext() if full_step else torch.no_grad()), torch.autocast('cuda', dtype=torch.bfloat16):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            model = initialize_model(**config, device=device)
            batch = get_random_batch(batch_size=args.batch_size, context_length=128, vocab_size=10_000, device=device)
            optimizer = AdamW(model.parameters(), lr=1e-3)
            loss = cross_entropy(model(batch[0]), batch[1])
            if full_step:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            torch.cuda.synchronize(device=device)
            peak_usage = torch.cuda.max_memory_allocated(device=device)
            print(f'[$2.7$B], [{"full step" if full_step else "forward"}], [{format_bytes(peak_usage)}],')
            del model, batch, optimizer, loss



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--d_model', type=int, default=768)
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--passes', type=str, default='both', choices=['forward', 'backward', 'both'])
    args = parser.parse_args()

    # (benchmarking_script)
    # perform_all_profiles(include_warmup=True, rms_norm=False)

    # (function_call_table)
    # run_pytorch_profile(enable_backward=True, enable_optimizer=True)

    # (benchmarking_mixed_precision)
    # test_mp(dtype=torch.float16)

    # (pytorch_layernorm)
    # norm_bench()

    # (rmsnorm_forward_benchmarking)
    # norm_bench(include_backward=True)

    # perform_all_profiles_norm(include_warmup=True, mixed_precision=False, compile=False, passes=['forward', 'both'])

    # (memory_profiling)
    # memory_profile(enable_backward=True, enable_optimizer=True, mixed_precision=False)

    peak_memory_usage()
    # peak_memory_usage_mixed()


