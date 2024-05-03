from datetime import timedelta
from typing import Literal
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from timeit import default_timer as timer
import numpy as np
from collections import defaultdict as dd

is_multinode = True

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
    dist.init_process_group(backend, timeout=timedelta(seconds=60))


def distributed_bench(rank, world_size, result_queue, backend, device_type='cpu', multinode=False):
    if is_multinode:
        setup_multinode(backend)
        rank, local_rank, world_size = int(os.environ['RANK']), int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE'])
        print(f'Rank: {rank}, Local Rank: {local_rank}, World Size: {world_size}')
    else:
        setup(rank, world_size, backend)
    if device_type == 'cuda':
        device = f'cuda:{rank}'
    else:
        device = 'cpu'
    for tensor_size in tensor_sizes:
        data = torch.randn((tensor_size,), device=device)
        dist.barrier()
        start = timer()
        dist.all_reduce(data, op=dist.ReduceOp.SUM, async_op=False)
        if device != 'cpu':
            torch.cuda.synchronize(device=device)
        end = timer()
        result_queue.send((tensor_size, end - start))

def run_bench(backend, device_type, world_size, multinode=False):
    parent_conn, child_conn = mp.Pipe()
    mp.spawn(fn=distributed_bench, nprocs=world_size, args=(world_size, child_conn, backend, device_type), join=True)
    results = [parent_conn.recv() for _ in range(world_size * len(tensor_sizes))]
    d = dd(list)
    for tensor_size, time in results:
        d[tensor_size].append(time)

    for tensor_size, times in d.items():
        mean_time = np.mean(times)
        print(f'[{backend.upper()} + {device_type.upper()}], [{format_bytes(tensor_size * 4)}], [${world_size}$], [${mean_time:.2e}$],')
    

def run_all_benches():
    for backend in ['gloo', 'nccl']:
        for device_type in ['cpu', 'cuda'] if backend == 'gloo' else ['cuda']:
            for world_size in 2, 4, 6:
                # print('running bench for', backend, device_type, tensor_size, world_size)
                run_bench(backend, device_type, world_size)  # Loops tensor_sizes within run


if __name__ == '__main__':
    run_all_benches()