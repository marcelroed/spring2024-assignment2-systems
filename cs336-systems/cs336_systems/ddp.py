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

def setup():
    dist.init_process_group(backend='nccl', timeout=timedelta(seconds=60))

def mprint(*args, **kwargs):
    if int(os.environ['RANK']) == 0:
        print(*args, **kwargs)

def ddp_train(do_comparison=False):
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

def main():
    ddp_train(do_comparison=False)


if __name__ == '__main__':
    main()

    
