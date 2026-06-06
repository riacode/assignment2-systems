from collections import namedtuple
import json
from pathlib import Path
import torch
from cs336_basics.flash_attention import FlashAttention2Triton
from cs336_basics.modal_utils import DATA_PATH, VOLUME_MOUNTS, app, build_image, user_volume
from cs336_basics.model import scaled_dot_product_attention

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = namedtuple("triton", ["jit", "make_block_ptr", "testing"])
    triton.jit = lambda x: x
    tl = namedtuple("tl", ["constexpr"])


SEQ_LENS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536] # sweep
D_MODELS = [16, 32, 64, 128]
PRECISIONS = {"bf16": torch.bfloat16, "fp32": torch.float32}

def _bench_forward(attention_fn, q, k, v, warmup, rep):
    # report a table that includes latencies for forward
    def forward():
        with torch.no_grad():
            attention_fn(q, k, v)
    return triton.testing.do_bench(forward, warmup=warmup, rep=rep)

def _bench_backward(attention_fn, q, k, v, dO, warmup, rep):
    # report a table that includes latencies for backward
    q.grad = None
    k.grad = None
    v.grad = None
    O = attention_fn(q, k, v)
    def backward():
        O.backward(dO, retain_graph=True)
    return triton.testing.do_bench(backward, warmup=warmup, rep=rep, grad_to_none=[q, k, v])

def _bench_forward_backward(attention_fn, q, k, v, dO, warmup, rep):
    # end-to-end forward-backward pass
    def forward_backward():
        q.grad = None
        k.grad = None
        v.grad = None
        O = attention_fn(q, k, v)
        O.backward(dO)
    return triton.testing.do_bench(forward_backward, warmup=warmup, rep=rep)

def _run_config(implementation, seq_len, d_model, precision_name, warmup, rep):
    device = "cuda"
    dtype = PRECISIONS[precision_name]
    # randomly generate any necessary inputs before benchmarking
    q = torch.randn(1, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(1, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(1, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
    dO = torch.randn(1, seq_len, d_model, device=device, dtype=dtype)

    # use batch size 1 and causal masking
    if implementation == "pytorch":
        mask = torch.arange(seq_len, device=device)[:, None] >= torch.arange(seq_len, device=device)[None, :]
        attention_fn = lambda Q, K, V: scaled_dot_product_attention(Q=Q, K=K, V=V, mask=mask)
    else:
        attention_fn = lambda Q, K, V: FlashAttention2Triton.apply(Q, K, V, True)

    forward_ms = _bench_forward(attention_fn, q, k, v, warmup, rep)
    backward_ms = _bench_backward(attention_fn, q, k, v, dO, warmup, rep)
    forward_backward_ms = _bench_forward_backward(attention_fn, q, k, v, dO, warmup, rep)

    return {
        "implementation": implementation,
        "batch_size": 1,
        "seq_len": seq_len,
        "d_model": d_model,
        "precision": precision_name,
        "status": "ok",
        "forward_ms": forward_ms,
        "backward_ms": backward_ms,
        "forward_backward_ms": forward_backward_ms,
    }

@app.function(image=build_image(), volumes=VOLUME_MOUNTS, gpu="B200", timeout=32400)
def run_flash_attention_benchmark_remote(warmup=10, rep=25):
    warmup = int(warmup)
    rep = int(rep)
    output_path = Path(DATA_PATH) / "flash_attention_benchmark.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for precision_name in PRECISIONS:
            for d_model in D_MODELS:
                for seq_len in SEQ_LENS:
                    for implementation in ["pytorch", "triton"]:
                        try:
                            row = _run_config(implementation, seq_len, d_model, precision_name, warmup, rep)
                        except torch.OutOfMemoryError as exc:
                            torch.cuda.empty_cache()
                            row = {
                                "implementation": implementation,
                                "batch_size": 1,
                                "seq_len": seq_len,
                                "d_model": d_model,
                                "precision": precision_name,
                                "status": "oom",
                                "error": str(exc)[:2000],
                            }

                        print(row)
                        f.write(json.dumps(row) + "\n")
                        f.flush()
    user_volume.commit()

@app.local_entrypoint()
def benchmark_flash_attention(warmup=10, rep=25):
    warmup = int(warmup)
    rep = int(rep)
    run_flash_attention_benchmark_remote.remote(warmup=warmup, rep=rep)
