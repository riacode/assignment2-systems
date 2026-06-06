import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from cs336_basics.ddp import DDP, DDP2
from cs336_basics.modal_utils import app, build_image
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

WORLD_SIZE = 2
BATCH_SIZE = 4
CONTEXT_LENGTH = 512
VOCAB_SIZE = 10000

def _training_step(ddp_model, optimizer, batch, targets):
    optimizer.zero_grad()
    logits = ddp_model(batch)
    loss = cross_entropy(logits, targets)
    loss.backward()
    torch.cuda.synchronize()
    start = time.perf_counter()
    ddp_model.finish_gradient_synchronization()
    torch.cuda.synchronize()
    end = time.perf_counter() - start
    optimizer.step()
    return end

def _worker(rank, ddp_type):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=WORLD_SIZE)
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"
    local_batch_size = BATCH_SIZE // WORLD_SIZE
    ddp_cls = DDP2 if ddp_type == "overlap" else DDP
    model = ddp_cls(
        BasicsTransformerLM(
            vocab_size=VOCAB_SIZE,
            context_length=CONTEXT_LENGTH,
            d_model=2560,
            num_layers=32,
            num_heads=32,
            d_ff=10240,
        ).to(device)
    )
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
    batch = torch.randint(0, VOCAB_SIZE, (local_batch_size, CONTEXT_LENGTH), device=device)
    targets = torch.randint(0, VOCAB_SIZE, (local_batch_size, CONTEXT_LENGTH), device=device)

    for _ in range(5):
        _training_step(model, optimizer, batch, targets)

    training_times = []
    communication_times = []
    for _ in range(10):
        torch.cuda.synchronize()
        start = time.perf_counter()
        end = _training_step(model, optimizer, batch, targets)
        torch.cuda.synchronize()
        training_times.append(time.perf_counter() - start)
        communication_times.append(end)

    if rank == 0:
        training_mean = sum(training_times) / len(training_times)
        communication_mean = sum(communication_times) / len(communication_times)
        row = {
            "model_size": "xl",
            "ddp_type": ddp_type,
            "batch_size": BATCH_SIZE,
            "world_size": WORLD_SIZE,
            "context_length": CONTEXT_LENGTH,
            "vocab_size": VOCAB_SIZE,
            "training_mean": training_mean,
            "communication_mean": communication_mean,
            "ratio": communication_mean / training_mean,
        }
        print(row)
    dist.destroy_process_group()

@app.function(image=build_image(), gpu="B200:2", timeout=1500)
def run_naive_ddp_benchmark_remote(ddp_type="flat"):
    mp.spawn(_worker, args=(ddp_type,), nprocs=WORLD_SIZE, join=True)

@app.local_entrypoint()
def benchmark_naive_ddp(ddp_type="flat"):
    run_naive_ddp_benchmark_remote.remote(ddp_type=ddp_type)
