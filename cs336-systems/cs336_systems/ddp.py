from typing import Any, Iterable, Type
from datetime import timedelta
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from timeit import default_timer as timer
from cs336_systems.profile_script import initialize_model, configs, get_random_batch
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from copy import deepcopy
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity
from torch.optim import Optimizer
from contextlib import nullcontext
from collections import defaultdict

def ceildiv(a, b):
    return -(-a // b)

def setup():
    dist.init_process_group(backend='nccl', timeout=timedelta(seconds=60))

def mprint(*args, **kwargs):
    if int(os.environ['RANK']) == 0:
        print(*args, **kwargs)

def ddp_train(do_comparison=False, flatten=True, prof=None):
    print('Setting up process group')
    setup()
    print('Starting training')
    batch_size = 2
    context_length = 128
    vocab_size = 10_000
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{local_rank}'
    torch.random.manual_seed(0)
    # torch.use_deterministic_algorithms(True)
    model = initialize_model(**configs['small'], context_length=context_length, vocab_size=vocab_size, device=device)
    batch = get_random_batch(batch_size, context_length, vocab_size=vocab_size, device=device)
    if do_comparison and rank == 0:
        one_model = deepcopy(model)
    dist.broadcast(batch[0], 0)
    dist.broadcast(batch[1], 0)

    local_batch_size = batch_size // world_size
    # local_batch_x = batch[0][:]
    # local_batch_y = batch[1][:]

    local_batch_x = batch[0][local_rank * local_batch_size:(local_rank + 1) * local_batch_size]
    local_batch_y = batch[1][local_rank * local_batch_size:(local_rank + 1) * local_batch_size]

    for param in model.parameters():
        dist.broadcast(param.data, 0)

    for model_name, config in configs.items():
        model = initialize_model(**config, context_length=context_length, vocab_size=vocab_size, device=device)
        optimizer = AdamW(model.parameters(), lr=1e-3)
        for i in range(5):
            if i == 4 and rank == 0:
                start = timer()
            optimizer.zero_grad()
            logits = model(local_batch_x)
            loss = cross_entropy(logits, local_batch_y)
            loss.backward()

            if flatten:
                grads = torch._utils._flatten_dense_tensors(tensors=[param.grad for param in model.parameters()])
                dist.all_reduce(grads, op=dist.ReduceOp.AVG, async_op=False)
                grads = torch._utils._unflatten_dense_tensors(flat=grads, tensors=[param.grad for param in model.parameters()])
                for param, grad in zip(model.parameters(), grads):
                    param.grad[:] = grad
            else:
                for param in model.parameters():
                    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, async_op=False)
            
            optimizer.step()
            torch.cuda.synchronize()
            if i == 4 and rank == 0:
                end = timer()
                print(f'[{model_name}], [{end - start:.2e}],')

        
        if do_comparison and rank == 0:
            # Train without ddp
            print('Training without DDP')
            optimizer = AdamW(one_model.parameters(), lr=1e-3)
            for i in range(100):
                optimizer.zero_grad()
                logits = one_model(batch[0])
                loss = cross_entropy(logits, batch[1])
                loss.backward()
                optimizer.step()
            for param, one_param in zip(model.parameters(), one_model.parameters()):
                torch.testing.assert_close(param, one_param)
            print('Model parameters are equal!')

    # torch.use_deterministic_algorithms(False)

def ddp_train_with_module(prof=None, use_configs=None):
    if isinstance(use_configs, str):
        use_configs = {use_configs: configs[use_configs]}
    else:
        use_configs = configs
    setup()
    batch_size = 2
    context_length = 128
    vocab_size = 10_000
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{local_rank}'
    torch.random.manual_seed(0)
    # torch.use_deterministic_algorithms(True)
    batch = get_random_batch(batch_size, context_length, vocab_size=vocab_size, device=device)

    dist.broadcast(batch[0], 0)
    dist.broadcast(batch[1], 0)

    local_batch_size = batch_size // world_size
    # local_batch_x = batch[0][:]
    # local_batch_y = batch[1][:]

    local_batch_x = batch[0][local_rank * local_batch_size:(local_rank + 1) * local_batch_size]
    local_batch_y = batch[1][local_rank * local_batch_size:(local_rank + 1) * local_batch_size]

    for model_name, config in use_configs.items():
        model = initialize_model(**config, context_length=context_length, vocab_size=vocab_size, device=device)
        model = DDP(model)
        optimizer = AdamW(model.parameters(), lr=1e-3)
        for i in range(5):
            if prof is not None:
                prof.step()
            if i == 4 and rank == 0:
                start = timer()
            optimizer.zero_grad()
            logits = model(local_batch_x)
            loss = cross_entropy(logits, local_batch_y)
            loss.backward()
            model.finish_gradient_synchronization()

            optimizer.step()
            torch.cuda.synchronize()
            if i == 4 and rank == 0:
                end = timer()
                print(f'[{model_name}], [{end - start:.2e}],')
        prof.step()


PROFILE = True
def main():
    # ddp_train(do_comparison=False, flatten=True)

    with (profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
        schedule=torch.profiler.schedule(wait=1, warmup=4, active=1),
        record_shapes=True,
        profile_memory=False,
        with_stack=True,
    ) if PROFILE else nullcontext()) as prof:
        prof.step()
        ddp_train_with_module(prof, 'xl')
    prof.export_stacks(f'stacks{os.environ["RANK"]}.txt', 'self_cuda_time_total')
    print(prof.key_averages().table(sort_by='cpu_time_total', row_limit=50))


class DDP(nn.Module):
    def __init__(self, module: torch.nn.Module):
        super(DDP, self).__init__()
        self.module = module
        self.back_handles = []
        self.post_acc_handles = []
        def transform_grad(param):
            with torch.no_grad():
                param.grad.data /= dist.get_world_size()
            handle = dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, async_op=True)
            self.back_handles.append(handle)

        for parameter in self.module.parameters():
            dist.broadcast(parameter.data, 0)
            if parameter.requires_grad:
                self.post_acc_handles.append(
                    parameter.register_post_accumulate_grad_hook(transform_grad)
                )

    def __del__(self):
        for handle in self.post_acc_handles:
            handle.remove()
    
    def forward(self, *inputs, **kwargs):
        return self.module.forward(*inputs, **kwargs)
    
    def finish_gradient_synchronization(self):
        while self.back_handles:
            self.back_handles.pop().wait()


class Bucket:
    def __init__(self, count: int):
        self.count = count
        self.params = []
    
    def add(self, param):
        self.params.append(param)
        if len(self.params) == self.count:
            result = self.perform_sync()
            self.params = []
            return result
        return None
    
    def perform_sync(self):
        flat_grad = torch._utils._flatten_dense_tensors(tensors=[p.grad for p in self.params])
        flat_grad /= dist.get_world_size()
        handle = dist.all_reduce(flat_grad, op=dist.ReduceOp.SUM, async_op=True)
        return (handle, self.params, flat_grad)
    
    @classmethod
    def from_params(cls, params: list[nn.Parameter]):
        bucket = cls(len(params))
        return bucket


class DDPBucketed(nn.Module):
    def __init__(self, module: nn.Module, bucket_size_mb: float):
        super(DDPBucketed, self).__init__()
        self.module = module
        self.bucket_size_mb = bucket_size_mb
        self.back_handles = []
        self.post_acc_handles = []
        self.param_to_bucket = {}
        self.all_buckets = []

        def transform_grad(param):
            maybe_handle = self.param_to_bucket[param].add(param)
            if maybe_handle is not None:
                self.back_handles.append(maybe_handle)
        
        bucked_size_b = bucket_size_mb * 2**20

        # How many bytes we have in the current bucket
        running_b = 0
        buckets = []
        current_bucket_params = []

        # Construct buckets for each parameter
        for parameter in reversed(list(self.module.parameters())):
            dist.broadcast(parameter.data, 0)
            if not parameter.requires_grad:
                continue

            parameter_bytes = parameter.data.nbytes
            if running_b + parameter_bytes > bucked_size_b and running_b > 0:
                # Create a new bucket
                saved_bucket = Bucket.from_params(current_bucket_params)
                for param in current_bucket_params:
                    self.param_to_bucket[param] = saved_bucket
                buckets.append(saved_bucket)
                current_bucket_params = []
                running_b = 0

            current_bucket_params.append(parameter)

            if parameter.requires_grad:
                self.post_acc_handles.append(
                    parameter.register_post_accumulate_grad_hook(transform_grad)
                )
        if current_bucket_params:
            saved_bucket = Bucket.from_params(current_bucket_params)
            for param in current_bucket_params:
                self.param_to_bucket[param] = saved_bucket
            buckets.append(saved_bucket)
        
    def forward(self, *inputs, **kwargs):
        return self.module.forward(*inputs, **kwargs)
    
    def finish_gradient_synchronization(self):
        while self.back_handles:
            handle, params, flat_grad = self.back_handles.pop(0)
            handle.wait()
            new_grads = torch._utils._unflatten_dense_tensors(flat_grad, params)
            for param, new_grad in zip(params, new_grads):
                param.grad.data[:] = new_grad

                
class ShardedOptimizer(Optimizer):
    def __init__(self, params: Iterable[nn.Parameter], optimizer_cls: Type[Optimizer], **kwargs: Any):
        params = list(params)
        self.optimizer = optimizer_cls(params, **kwargs)
        self.param_group_shard_size = {}
        self.belongs_to_shard = {}
        super(ShardedOptimizer, self).__init__(params, {})

    def step(self, closure=None, **kwargs):
        self.optimizer.step(closure=closure, **kwargs)
        self._synchronize_params()
    
    def _synchronize_params(self):
        for param, rank in self.belongs_to_shard.items():
            dist.broadcast(param.data, rank)

    def add_param_group(self, param_group: dict[str, Any]):
        n_shards = dist.get_world_size()
        rank = dist.get_rank()
        shard_param_group = defaultdict(list)
        for group_name, params in param_group.items():
            group_size_bytes = sum(p.data.nbytes for p in params)
            shard_size = ceildiv(group_size_bytes, n_shards)
            self.param_group_shard_size[group_name] = shard_size

            current_start_byte = 0
            for p in params:
                for_rank = current_start_byte // shard_size
                assert 0 <= for_rank < n_shards
                self.belongs_to_shard[p] = for_rank
                if rank == for_rank:
                    shard_param_group[group_name].append(p)
                current_start_byte += p.data.nbytes
        super(ShardedOptimizer, self).add_param_group(shard_param_group)


if __name__ == '__main__':
    main()

    
