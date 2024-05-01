from typing import Literal
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from timeit import default_timer as timer
import numpy as np

def format_bytes(n_bytes):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if n_bytes < 1000:
            return f'${n_bytes:.0f}$ {unit}'
        n_bytes /= 1000
    return f'${n_bytes:.0f}$ TB'

def setup(rank, world_size, backend: Literal['gloo', 'nccl'] = 'gloo'):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def distributed_bench(rank, world_size, result_queue, tensor_size, backend, device_type='cpu'):
    setup(rank, world_size, backend)
    if device_type == 'cuda':
        device = f'cuda:{rank}'
    else:
        device = 'cpu'
    data = torch.randn((tensor_size,), device=device)
    dist.barrier()
    start = timer()
    dist.all_reduce(data, op=dist.ReduceOp.SUM, async_op=False)
    if device != 'cpu':
        torch.cuda.synchronize(device=device)
    end = timer()
    result_queue.send(end - start)

def run_bench(backend, device_type, tensor_size, world_size):
    parent_conn, child_conn = mp.Pipe()
    mp.spawn(fn=distributed_bench, nprocs=world_size, args=(world_size, child_conn, tensor_size, backend, device_type), join=True)
    results = [parent_conn.recv() for _ in range(world_size)]
    mean_time = np.mean(results)
    print(f'[{backend.upper()} + {device_type.upper()}], [{format_bytes(tensor_size * 4)}], [${world_size}$], [${mean_time:.2e}$],')
    

def run_all_benches():
    tensor_sizes = np.array([512_000, 1_000_000, 10_000_000, 50_000_000, 100_000_000, 500_000_000, 1_000_000_000]) // 4
                    
    for backend in ['gloo', 'nccl']:
        for device_type in ['cpu', 'cuda'] if backend == 'gloo' else ['cuda']:
            for tensor_size in tensor_sizes:
                for world_size in 2, 4, 6:
                    # print('running bench for', backend, device_type, tensor_size, world_size)
                    run_bench(backend, device_type, tensor_size, world_size)


if __name__ == '__main__':
    run_all_benches()

