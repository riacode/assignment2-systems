import json
import statistics
import timeit
from pathlib import Path
import torch
from cs336_basics.modal_utils import DATA_PATH, VOLUME_MOUNTS, app, build_image, user_volume
from cs336_basics.model import scaled_dot_product_attention

D_MODELS = [16, 32, 64, 128]
SEQ_LENS = [256, 1024, 4096, 8192, 16384, 32768]

def _time_forward(attention_fn, q, k, v, num_steps):
    times = []
    with torch.no_grad():
        for _ in range(num_steps):
            start = timeit.default_timer()
            attention_fn(q, k, v)
            torch.cuda.synchronize()
            times.append(timeit.default_timer() - start)
    return times

def _time_backward(attention_fn, q, k, v, num_steps):
    times = []
    for _ in range(num_steps):
        q.grad = None
        k.grad = None
        v.grad = None
        output = attention_fn(q, k, v)
        loss = output.sum()
        start = timeit.default_timer()
        loss.backward()
        torch.cuda.synchronize()
        times.append(timeit.default_timer() - start)
    return times

def run_one_config(d_model, seq_len, num_steps, warmup_steps, compiled=False):
    device = "cuda"
    batch_size = 8
    q = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
    k = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
    v = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
    attention_fn = torch.compile(scaled_dot_product_attention) if compiled else scaled_dot_product_attention
    for _ in range(warmup_steps):
        output = attention_fn(q, k, v)
        output.sum().backward()
        q.grad = None
        k.grad = None
        v.grad = None
        torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    forward_times = _time_forward(attention_fn, q, k, v, num_steps)
    output = attention_fn(q, k, v)
    torch.cuda.synchronize()
    memory_before_backward_bytes = torch.cuda.memory_allocated()

    torch.cuda.reset_peak_memory_stats()
    backward_times = _time_backward(attention_fn, q, k, v, num_steps)

    return {
        "batch_size": batch_size,
        "compiled": compiled,
        "d_model": d_model,
        "seq_len": seq_len,
        "status": "ok",
        "forward_mean": statistics.mean(forward_times),
        "forward_std": statistics.stdev(forward_times) if len(forward_times) > 1 else None,
        "backward_mean": statistics.mean(backward_times),
        "backward_std": statistics.stdev(backward_times) if len(backward_times) > 1 else None,
        "memory_before_backward": memory_before_backward_bytes / (1024**3),
        "peak_backward_memory": torch.cuda.max_memory_allocated() / (1024**3),
    }


@app.function(
    image=build_image(),
    volumes=VOLUME_MOUNTS,
    gpu="B200",
    timeout=32400,
    retries=0,
)
def run_attention_benchmark_remote(num_steps=100, warmup_steps=5):
    output_path = Path(DATA_PATH) / "pytorch_attention_benchmark.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = []

    with output_path.open("w", encoding="utf-8") as f:
        for d_model in D_MODELS:
            for seq_len in SEQ_LENS:
                try:
                    row = run_one_config(d_model, seq_len, num_steps, warmup_steps)
                except torch.OutOfMemoryError as exc:
                    torch.cuda.empty_cache()
                    row = {
                        "batch_size": 8,
                        "d_model": d_model,
                        "seq_len": seq_len,
                        "status": "oom",
                        "error": str(exc),
                    }
                print(row)
                f.write(json.dumps(row) + "\n")
                f.flush()
                results.append(row)

    user_volume.commit()
    return results


@app.function(
    image=build_image(),
    volumes=VOLUME_MOUNTS,
    gpu="B200",
    timeout=32400,
    retries=0,
)
def run_compiled_attention_benchmark_remote(num_steps=100, warmup_steps=5):
    output_path = Path(DATA_PATH) / "torch_compile_attention_benchmark.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = []

    with output_path.open("w", encoding="utf-8") as f:
        for d_model in D_MODELS:
            for seq_len in SEQ_LENS:
                for compiled in [False, True]:
                    try:
                        row = run_one_config(d_model, seq_len, num_steps, warmup_steps, compiled=compiled)
                    except torch.OutOfMemoryError as exc:
                        torch.cuda.empty_cache()
                        row = {
                            "batch_size": 8,
                            "d_model": d_model,
                            "seq_len": seq_len,
                            "compiled": compiled,
                            "status": "oom",
                            "error": str(exc),
                        }
                    print(row)
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    results.append(row)

    user_volume.commit()
    return results


@app.local_entrypoint()
def main(num_steps=100, warmup_steps=5):
    run_attention_benchmark_remote.remote(num_steps=num_steps, warmup_steps=warmup_steps)


@app.local_entrypoint()
def benchmark_torch_compile_attention(num_steps=100, warmup_steps=5):
    run_compiled_attention_benchmark_remote.remote(num_steps=num_steps, warmup_steps=warmup_steps)
