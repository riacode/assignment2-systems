import os
from collections import namedtuple
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from cs336_basics.fsdp import FSDP
from cs336_basics.modal_utils import app, build_image
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
try:
    import triton
    import triton.language as tl
except ImportError:
    triton = namedtuple("triton", ["jit", "make_block_ptr", "testing"])
    triton.jit = lambda x: x
    tl = namedtuple("tl", ["constexpr"])


class Config:
    ctx_len = 32768
    vocab_size = 151936
    d_model = 4096
    d_ff = 11008
    num_layers = 34
    num_heads = 32
    torch_dtype = torch.bfloat16
    is_causal = True
    batch_size = 2

cfg = Config()
WORLD_SIZE = 2

def _worker(rank, warmup, rep):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=WORLD_SIZE)
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"
    labels, targets = torch.randint(high=cfg.vocab_size, size=(2, cfg.batch_size // WORLD_SIZE, cfg.ctx_len), device=device)
    model = FSDP(
        BasicsTransformerLM(
            vocab_size=cfg.vocab_size,
            context_length=cfg.ctx_len,
            d_model=cfg.d_model,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            d_ff=cfg.d_ff,
        ).to(device=device, dtype=cfg.torch_dtype),
        compute_dtype=cfg.torch_dtype,
    )
    optimizer = AdamW(model.parameters())

    def train_step():
        optimizer.zero_grad(set_to_none=True)
        res = model(labels)
        loss = cross_entropy(res, targets).sum()
        loss.backward()
        model.finish_gradient_synchronization()
        optimizer.step()

    timing_results = triton.testing.do_bench(train_step, rep=rep, warmup=warmup)
    if rank == 0:
        print(timing_results)
    dist.destroy_process_group()

@app.function(image=build_image(), gpu="B200:2", timeout=3600)
def run_leaderboard_benchmark_remote(warmup=1, rep=3):
    mp.spawn(_worker, args=(warmup, rep), nprocs=WORLD_SIZE, join=True)

@app.local_entrypoint()
def leaderboard_benchmark(warmup=1, rep=3):
    run_leaderboard_benchmark_remote.remote(warmup, rep)
