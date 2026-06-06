import json
import os
import timeit
import statistics
from pathlib import Path
import shlex
import shutil
import subprocess
from contextlib import nullcontext
import torch
import torch.cuda
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.memory_profile import run_memory_profile

# add modal usage
from cs336_basics.modal_utils import DATA_PATH, VOLUME_MOUNTS, app, build_image, build_nsys_profile_image, user_volume
BENCHMARK_JSONL = Path(DATA_PATH) / "benchmark_runs_warmup2.jsonl"

def _append_benchmark_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")

# define model sizes
MODEL_SIZES = {
    "small": {
        "d_model": 768,
        "d_ff": 3072,
        "num_layers": 12,
        "num_heads": 12,
    },
    "medium": {
        "d_model": 1024,
        "d_ff": 4096,
        "num_layers": 24,
        "num_heads": 16,
    },
    "large": {
        "d_model": 1280,
        "d_ff": 5120,
        "num_layers": 36,
        "num_heads": 20,
    },
    "xl": {
        "d_model": 2560,
        "d_ff": 10240,
        "num_layers": 32,
        "num_heads": 32,
    },
    "10B": {
        "d_model": 4608,
        "d_ff": 12288,
        "num_layers": 50,
        "num_heads": 36,
    },
}

def _mixed_precision(use_bf16):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()

def forward_pass(model, input, use_bf16=False):
    with torch.inference_mode(), _mixed_precision(use_bf16):
        model.forward(input)
    torch.cuda.synchronize()

def forward_backward_pass(model, input, targets, use_bf16=False):
    with _mixed_precision(use_bf16):
        output = model.forward(input)
        loss = cross_entropy(output, targets)
    loss.backward()
    torch.cuda.synchronize()

def full_step(model, input, targets, optimizer, use_bf16=False):
    optimizer.zero_grad()
    with _mixed_precision(use_bf16):
        output = model.forward(input)
        loss = cross_entropy(output, targets)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()

def initialize_model(vocab_size, context_length, d_model, num_layers, num_heads, d_ff):
    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
    )
    return model

def warmup_steps(model, num_steps, input, mode, optimizer, targets, use_bf16=False):
    if mode == "forward":
        for _ in range(num_steps):
            forward_pass(model, input, use_bf16)
    elif mode == "forward_backward":
        for _ in range(num_steps):
            forward_backward_pass(model, input, targets, use_bf16)
    elif mode == "forward_backward_optimizer":
        for _ in range(num_steps):
            full_step(model, input, targets, optimizer, use_bf16)

def benchmark_steps(model, num_steps, batch, mode, optimizer, targets, use_bf16=False):
    if mode == "forward":
        print(f"Benchmarking {num_steps} forward steps")
        forward_pass_times = []
        for _ in range(num_steps):
            start_time = timeit.default_timer()
            forward_pass(model, batch, use_bf16)
            end_time = timeit.default_timer()
            forward_pass_times.append(end_time - start_time)
            print(f"Forward pass time: {end_time - start_time} seconds")
        print(f"Average forward pass time: {sum(forward_pass_times) / len(forward_pass_times)} seconds")
        print(f"Standard deviation of forward pass time: {statistics.stdev(forward_pass_times)} seconds")
        return {"step_times": forward_pass_times}
    if mode == "forward_backward":
        print(f"Benchmarking {num_steps} forward and backward steps")
        forward_backward_pass_times = []
        for _ in range(num_steps):
            start_time = timeit.default_timer()
            forward_backward_pass(model, batch, targets, use_bf16)
            end_time = timeit.default_timer()
            forward_backward_pass_times.append(end_time - start_time)
            print(f"Forward and backward pass time: {end_time - start_time} seconds")
        print(f"Average forward and backward pass time: {sum(forward_backward_pass_times) / len(forward_backward_pass_times)} seconds")
        print(f"Standard deviation of forward and backward pass time: {statistics.stdev(forward_backward_pass_times)} seconds")
        return {"step_times": forward_backward_pass_times}
    if mode == "forward_backward_optimizer":
        print(f"Benchmarking {num_steps} forward and backward and optimizer steps")
        full_step_times = []
        for _ in range(num_steps):
            start_time = timeit.default_timer()
            full_step(model, batch, targets, optimizer, use_bf16)
            end_time = timeit.default_timer()
            full_step_times.append(end_time - start_time)
            print(f"Full step time: {end_time - start_time} seconds")
        print(f"Average full step time: {sum(full_step_times) / len(full_step_times)} seconds")
        print(f"Standard deviation of full step time: {statistics.stdev(full_step_times)} seconds")
        return {"step_times": full_step_times}

@app.function(image=build_image(), volumes=VOLUME_MOUNTS, gpu="B200", max_containers=3, retries=0)
def run_benchmark_remote(config):
    row = {k: list(v) if k == "betas" and isinstance(v, tuple) else v for k, v in config.items()}
    row["status"] = "ok"
    row["step_times"] = None
    row["mean_time"] = None
    row["std_time"] = None
    row["error"] = None

    model_size = MODEL_SIZES[config["model_size"]]
    mode = config["mode"]
    vocab_size = config["vocab_size"]
    batch_size = config["batch_size"]
    context_length = config["context_length"]
    num_steps = config["num_steps"]
    num_warmup_steps = config["num_warmup_steps"]
    lr = config["lr"]
    weight_decay = config["weight_decay"]
    betas = config["betas"]
    eps = config["eps"]
    use_bf16 = config.get("use_bf16", False)
    compile_model = config.get("compile_model", False)
    row["precision"] = "bf16" if use_bf16 else "fp32"
    row["compiled"] = compile_model
    device = "cuda"
    print(f"Running benchmark with config: {config}")
    try:
        model = initialize_model(vocab_size, context_length, model_size["d_model"], model_size["num_layers"], model_size["num_heads"], model_size["d_ff"]).to(device)
        if compile_model:
            model = torch.compile(model)
        batch = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
        targets = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
        warmup_steps(model, num_warmup_steps, batch, mode, optimizer, targets, use_bf16)
        row.update(benchmark_steps(model, num_steps, batch, mode, optimizer, targets, use_bf16))
        times = row["step_times"]
        row["mean_time"] = sum(times) / len(times)
        row["std_time"] = statistics.stdev(times) if len(times) > 1 else None
    except torch.OutOfMemoryError as exc:
        row["status"] = "oom"
        row["error"] = str(exc)[:2000]
        print(f"OOM: {row['error']}")
    return row

# Nsight profile presets
_NSYS_PROFILE_VERSIONS = {
    1: (
        "profile_result_v1",
        "profile_result_v1.nsys-rep",
        "--pytorch autograd-nvtx --gpu-metrics-devices all",
    ),
    2: (
        "profile_result_v2",
        "profile_result_v2.nsys-rep",
        "--trace=cuda,cudnn,cublas,osrt,nvtx "
        "--pytorch=functions-trace,autograd-shapes-nvtx "
        "--cudabacktrace=all "
        "--python-backtrace=cuda "
        "--gpu-metrics-devices=0",
    ),
    3: (
        "profile_result_v3",
        "profile_result_v3.nsys-rep",
        "--trace=cuda,cudnn,cublas,osrt,nvtx "
        "--pytorch=functions-trace,autograd-shapes-nvtx "
        "--cuda-memory-usage=true "
        "--gpu-metrics-devices=0",
    ),
}

PROFILING_ENTRY = "/profiling/benchmark.py"

def _nsys_report_stem_to_volume(tmp_stem, vol_filename):
    tmp_rep = Path(f"/tmp/{tmp_stem}.nsys-rep")
    vol_path = Path("/root") / DATA_PATH / vol_filename
    if tmp_rep.is_file():
        vol_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(tmp_rep, vol_path)
    user_volume.commit()
    return str(vol_path if vol_path.is_file() else tmp_rep)

@app.function(
    image=build_nsys_profile_image(),
    volumes=VOLUME_MOUNTS,
    gpu="B200",
    timeout=32400,
    max_containers=3,
    retries=0,
)
def check_caps_profile_with_args(version=1, profile_args=None, output_stem=None):
    return _run_nsys_profile_subprocess(version=version, profile_args=profile_args, output_stem=output_stem)

def _run_nsys_profile_subprocess(version: int, profile_args: list[str] | None, output_stem: str | None):
    stem, vol_file, extra_args = _NSYS_PROFILE_VERSIONS[version]
    if output_stem is not None:
        stem = output_stem
        vol_file = f"{output_stem}.nsys-rep"

    profile_args = profile_args or []
    env = os.environ.copy()
    if version == 3:
        env["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"
        env["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"

    command = ["nsys", "profile", "-o", f"/tmp/{stem}", *shlex.split(extra_args), "--", "python", PROFILING_ENTRY, *profile_args]
    subprocess.run(command, check=True, env=env)
    return _nsys_report_stem_to_volume(stem, vol_file)

def _run_modal_nsys_profile(version: int, profile_args: list[str] | None = None, output_stem: str | None = None) -> None:
    path = check_caps_profile_with_args.remote(version=version, profile_args=profile_args or [], output_stem=output_stem)
    vol_file = f"{output_stem}.nsys-rep" if output_stem is not None else _NSYS_PROFILE_VERSIONS[version][1]
    print(path)
    print(vol_file)

@app.function(
    image=build_image(),
    volumes=VOLUME_MOUNTS,
    gpu="B200",
    timeout=32400,
    retries=0,
)
def run_memory_profile_remote(config: dict):
    context_length = config["context_length"]
    mode = config["mode"]
    use_bf16 = config["use_bf16"]
    precision = "bf16" if use_bf16 else "fp32"
    batch_size = config.get("batch_size", 4)
    vocab_size = config.get("vocab_size", 10000)
    num_warmup_steps = config.get("num_warmup_steps", 1)
    checkpoint_subset_size = config.get("checkpoint_subset_size", 0)

    checkpoint_tag = f"_ckpt{checkpoint_subset_size}" if checkpoint_subset_size else ""
    snapshot_name = f"xl_ctx{context_length}_{mode}_{precision}{checkpoint_tag}.pickle"
    snapshot_path = Path("/root") / DATA_PATH / "memory_snapshots" / snapshot_name

    row = run_memory_profile(
        context_length=context_length,
        mode=mode,
        use_bf16=use_bf16,
        snapshot_path=snapshot_path,
        batch_size=batch_size,
        vocab_size=vocab_size,
        num_warmup_steps=num_warmup_steps,
        checkpoint_subset_size=checkpoint_subset_size,
    )
    user_volume.commit()
    row["snapshot_path"] = str(Path("memory_snapshots") / snapshot_name)
    return row

@app.local_entrypoint()
def memory_profile(
    context_length=2048,
    mode="training_step",
    use_bf16=False,
    batch_size=4,
    num_warmup_steps=1,
    checkpoint_subset_size=0,
):
    config = {
        "context_length": context_length,
        "mode": mode,
        "use_bf16": use_bf16,
        "batch_size": batch_size,
        "num_warmup_steps": num_warmup_steps,
        "checkpoint_subset_size": checkpoint_subset_size,
    }
    result = run_memory_profile_remote.remote(config)
    print(result)
    if result.get("snapshot_path"):
        print(result['snapshot_path'])

@app.local_entrypoint()
def profile_nsys(
    version=2,
    model_size="small",
    context_length=512,
    mode="forward",
    warmup_steps=2,
    profile_steps=1,
):
    profile_args = [
        "--model-size",
        model_size,
        "--context-length",
        str(context_length),
        "--mode",
        mode,
        "--warmup-steps",
        str(warmup_steps),
        "--profile-steps",
        str(profile_steps),
    ]
    output_stem = f"nsys_reports/nsys_{model_size}_{context_length}_{mode}_v{version}"
    _run_modal_nsys_profile(version, profile_args=profile_args, output_stem=output_stem)

@app.local_entrypoint()
def benchmark_mixed_precision(
    context_length=512,
    batch_size=4,
    num_steps=10,
    num_warmup_steps=2,
):
    all_configs = []
    for size in MODEL_SIZES:
        for mode in ["forward", "forward_backward"]:
            for use_bf16 in [False, True]:
                config = {
                    "model_size": size,
                    "mode": mode,
                    "use_bf16": use_bf16,
                    "vocab_size": 10000,
                    "batch_size": batch_size,
                    "context_length": context_length,
                    "num_steps": num_steps,
                    "num_warmup_steps": num_warmup_steps,
                    "lr": 1e-3,
                    "weight_decay": 0.01,
                    "betas": (0.9, 0.999),
                    "eps": 1e-8,
                }
                all_configs.append(config)

    results = run_benchmark_remote.map(all_configs)
    output_path = Path(DATA_PATH) / "benchmark_mixed_precision.jsonl"
    for result in results:
        print(result)
        _append_benchmark_jsonl(output_path, result)


@app.local_entrypoint()
def benchmark_torch_compile_transformer(
    context_length=512,
    batch_size=4,
    num_steps=10,
    num_warmup_steps=2,
):
    all_configs = []
    for size in MODEL_SIZES:
        for mode in ["forward", "forward_backward", "forward_backward_optimizer"]:
            for compile_model in [False, True]:
                config = {
                    "model_size": size,
                    "mode": mode,
                    "compile_model": compile_model,
                    "vocab_size": 10000,
                    "batch_size": batch_size,
                    "context_length": context_length,
                    "num_steps": num_steps,
                    "num_warmup_steps": num_warmup_steps,
                    "lr": 1e-3,
                    "weight_decay": 0.01,
                    "betas": (0.9, 0.999),
                    "eps": 1e-8,
                }
                all_configs.append(config)

    output_path = Path(DATA_PATH) / "torch_compile_transformer_benchmark.jsonl"
    results = run_benchmark_remote.map(all_configs)
    for result in results:
        print(result)
        _append_benchmark_jsonl(output_path, result)


@app.local_entrypoint()
def main():
    # 15 runs
    all_configs = []
    for size in MODEL_SIZES:
        for mode in ["forward", "forward_backward", "forward_backward_optimizer"]:
            config = {
                "model_size": size,
                "mode": mode,
                "vocab_size": 10000,
                "batch_size": 4,
                "context_length": 512,
                "num_steps": 10,
                "num_warmup_steps": 2,
                "lr": 1e-3,
                "weight_decay": 0.01,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
            }
            all_configs.append(config)

    BENCHMARK_JSONL.parent.mkdir(parents=True, exist_ok=True)
    results = run_benchmark_remote.map(all_configs)
    for result in results:
        print(result)
        _append_benchmark_jsonl(BENCHMARK_JSONL, result)
