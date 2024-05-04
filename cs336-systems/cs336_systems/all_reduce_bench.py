from datetime import timedelta
from typing import Literal
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from timeit import default_timer as timer
import numpy as np
from collections import defaultdict as dd


# First is for warmup
tensor_sizes = np.array([256_000, 512_000, 1_000_000, 10_000_000, 50_000_000, 100_000_000, 500_000_000, 1_000_000_000]) // 4

def format_bytes(n_bytes):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if n_bytes < 1000:
            return f'${n_bytes:.0f}$ {unit}'
        n_bytes /= 1000
    return f'${n_bytes:.0f}$ TB'

def setup(rank, world_size: int, backend: Literal['gloo', 'nccl'] = 'gloo'):
    if not is_multinode:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def setup_multinode(backend: Literal['gloo', 'nccl']):
    print(f'Initializing backend {backend} multinode')
    dist.init_process_group(backend, timeout=timedelta(seconds=60))


def distributed_bench(rank, world_size, backend, device_type='cpu'):
    if is_multinode:
        # print(f'Setting up with {os.environ}')
        rank, local_rank, world_size = int(os.environ['RANK']), int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE'])
        print(f'Rank: {rank}, Local Rank: {local_rank}, World Size: {world_size}')
    else:
        setup(rank, world_size, backend)
    if device_type == 'cuda':
        device = f'cuda:{rank}'
    else:
        device = 'cpu'
    for i, tensor_size in enumerate(tensor_sizes):
        data = torch.randn((tensor_size,), device=device)
        if device != 'cpu':
            torch.cuda.synchronize()
        dist.barrier()
        start = timer()
        dist.all_reduce(data, op=dist.ReduceOp.SUM, async_op=False)
        if device != 'cpu':
            torch.cuda.synchronize(device=device)
        end = timer()
        mean_time = torch.tensor(end - start, device=device)
        dist.all_reduce(mean_time, op=dist.ReduceOp.SUM, async_op=False)
        mean_time /= world_size
        if rank == 0:
            print(f'[{backend.upper()} + {device_type.upper()}], [{format_bytes(tensor_size * 4)}], [${world_size}$], [${mean_time:.2e}$],')

def run_bench(backend, device_type, world_size):
    mp.spawn(fn=distributed_bench, nprocs=world_size, args=(world_size, backend, device_type), join=True)

def run_bench_multinode(backend, device_type):
    rank = os.environ['RANK']
    world_size = os.environ['WORLD_SIZE']
    distributed_bench(rank, world_size, backend, device_type)

def run_all_benches():
    for backend in ['gloo', 'nccl']:
        for device_type in ['cpu', 'cuda'] if backend == 'gloo' else ['cuda']:
            for world_size in 2, 4, 6:
                # print('running bench for', backend, device_type, tensor_size, world_size)
                run_bench(backend, device_type, world_size)  # Loops tensor_sizes within run

def run_all_benches_multinode(backend='nccl'):
    setup_multinode(backend)
    for device_type in ['cpu', 'cuda'] if backend == 'gloo' else ['cuda']:
        run_bench_multinode(backend, device_type)


def main():
    # Run this with torchrun!
    print('Running benches')
    run_all_benches_multinode(backend='nccl')


if __name__ == '__main__':
    is_multinode = True
    main()