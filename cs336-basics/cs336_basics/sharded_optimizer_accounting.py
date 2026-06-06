import os
import time
import torch
import torch.distributed as dist
from cs336_basics.modal_utils import app, build_image
import torch.multiprocessing as mp
from cs336_basics.ddp import DDP
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_basics.sharded_optimizer import ShardedOptimizer

WORLD_SIZE = 2
BATCH_SIZE = 4
CONTEXT_LENGTH = 512
VOCAB_SIZE = 10000

def _worker(rank, opt):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=WORLD_SIZE)
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"
    ddp_model = DDP(
        BasicsTransformerLM(
            vocab_size=VOCAB_SIZE,
            context_length=CONTEXT_LENGTH,
            d_model=2560,
            num_layers=32,
            num_heads=32,
            d_ff=10240,
        ).to(device)
    )
    torch.cuda.synchronize()
    peak_after_init = torch.cuda.max_memory_allocated() / 1024**3
    if opt == "sharded":
        optimizer = ShardedOptimizer(ddp_model.parameters(), AdamW, lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
    else:
        optimizer = AdamW(ddp_model.parameters(), lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)

    batch = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE // WORLD_SIZE, CONTEXT_LENGTH), device=device)
    targets = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE // WORLD_SIZE, CONTEXT_LENGTH), device=device)
    torch.cuda.synchronize()
    start = time.perf_counter()
    optimizer.zero_grad()
    logits = ddp_model(batch)
    loss = cross_entropy(logits, targets)
    loss.backward()
    ddp_model.finish_gradient_synchronization()
    torch.cuda.synchronize()
    peak_before_opt = torch.cuda.max_memory_allocated() / 1024**3
    optimizer.step()
    torch.cuda.synchronize()
    peak_after_opt = torch.cuda.max_memory_allocated() / 1024**3
    step_time = time.perf_counter() - start

    if rank == 0: # only first one
        actual_opt = optimizer
        if isinstance(optimizer, ShardedOptimizer):
            actual_opt = optimizer.optimizer
        optimizer_state_mem = 0
        for state in actual_opt.state.values():
            for value in state.values():
                if torch.is_tensor(value):
                    optimizer_state_mem += value.numel() * value.element_size()
        parameter_mem = sum(parameter.numel() * parameter.element_size() for parameter in ddp_model.parameters()) / 1024**3
        print(
            {
            "optimizer": opt,
            "peak_after_init": peak_after_init,
            "peak_before_opt": peak_before_opt,
            "peak_after_opt": peak_after_opt,
            "optimizer_state_mem": optimizer_state_mem/1024**3,
            "parameter_mem": parameter_mem,
            "step_time": step_time,
        }
        )
    dist.destroy_process_group()

@app.function(image=build_image(), gpu="B200:2", timeout=1500)
def run_sharded_optimizer_accounting_remote():
    for optimizer in ["unsharded", "sharded"]:
        mp.spawn(_worker, args=(optimizer,), nprocs=WORLD_SIZE, join=True)

@app.local_entrypoint()
def sharded_optimizer_accounting():
    run_sharded_optimizer_accounting_remote.remote()
