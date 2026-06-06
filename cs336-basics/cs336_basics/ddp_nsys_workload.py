import argparse
import os
import torch
import torch.cuda.nvtx as nvtx
import torch.distributed as dist
import torch.multiprocessing as mp
from cs336_basics.ddp import DDP, DDP2
from cs336_basics.fsdp import FSDP
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

WORLD_SIZE = 2
BATCH_SIZE = 4
CONTEXT_LENGTH = 512
VOCAB_SIZE = 10000

def _training_step(model, optimizer, batch, targets):
    optimizer.zero_grad()
    with nvtx.range("forward"):
        logits = model(batch)
        loss = cross_entropy(logits, targets)
    with nvtx.range("backward"):
        loss.backward()
    with nvtx.range("gradient_sync"):
        model.finish_gradient_synchronization()
    with nvtx.range("optimizer_step"):
        optimizer.step()

def _worker(rank, profile_type):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=WORLD_SIZE)
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"
    local_batch_size = BATCH_SIZE // WORLD_SIZE
    model = BasicsTransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        d_model=2560,
        num_layers=32,
        num_heads=32,
        d_ff=10240,
    ).to(device)
    model = (FSDP(model) if profile_type == "fsdp" else (DDP2 if profile_type == "overlap" else DDP)(model))
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
    batch = torch.randint(0, VOCAB_SIZE, (local_batch_size, CONTEXT_LENGTH), device=device)
    targets = torch.randint(0, VOCAB_SIZE, (local_batch_size, CONTEXT_LENGTH), device=device)
    _training_step(model, optimizer, batch, targets)
    torch.cuda.synchronize()
    with nvtx.range(f"{profile_type}_profile_step"):
        _training_step(model, optimizer, batch, targets)
    torch.cuda.synchronize()
    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-type", choices=["initial", "overlap", "fsdp"], required=True)
    args = parser.parse_args()
    mp.spawn(_worker, args=(args.profile_type,), nprocs=WORLD_SIZE, join=True)

if __name__ == "__main__":
    main()
