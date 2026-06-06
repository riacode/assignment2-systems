import json
import os
import time
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from cs336_basics.modal_utils import DATA_PATH, VOLUME_MOUNTS, app, build_image, user_volume

WORLD_SIZES = [2, 4, 6]
DATA_SIZES = [("1MB", 1), ("10MB", 10), ("100MB", 100), ("1GB", 1024)]

def _worker(rank, world_size, size_label, size_mb, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size) # use NCCL for distributed GPU training
    torch.cuda.set_device(rank) # different ranks use different GPUs
    data = torch.ones(size_mb * 1024 * 1024 // 4, device=f"cuda:{rank}", dtype=torch.float32)

    # warmup
    for _ in range(5):
        dist.all_reduce(data, async_op=False)
    torch.cuda.synchronize()

    times = []
    for _ in range(20):
        start = time.perf_counter() # benchmark the runtime
        dist.all_reduce(data, async_op=False)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, times) # collect

    if rank == 0: # write the row
        rank_means = [sum(times) / len(times) for times in gathered]
        row = {
            "world_size": world_size,
            "data_size": size_label,
            "data_size_mb": size_mb,
            "mean_ms": sum(rank_means) / len(rank_means),
        }
        output_path = Path(DATA_PATH) / "distributed_communication_benchmark.jsonl"
        with output_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(row)

    dist.destroy_process_group()

@app.function(image=build_image(), volumes=VOLUME_MOUNTS, gpu="B200:6", timeout=1500)
def run_distributed_communication_benchmark_remote():
    jsonl_path = Path(DATA_PATH) / "distributed_communication_benchmark.jsonl"
    jsonl_path.write_text("")
    for world_size in WORLD_SIZES:
        for size_label, size_mb in DATA_SIZES:
            mp.spawn(_worker, args=(world_size, size_label, size_mb, "29500"), nprocs=world_size, join=True)
    user_volume.commit()

@app.local_entrypoint()
def benchmark_distributed_communication():
    run_distributed_communication_benchmark_remote.remote()
