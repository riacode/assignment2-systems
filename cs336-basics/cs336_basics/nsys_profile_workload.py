from __future__ import annotations
import cs336_basics.model as model_impl
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy, softmax
from cs336_basics.optimizer import AdamW
import argparse
import math
import time
from collections.abc import Callable
import torch
import torch.cuda.nvtx as nvtx
from einops import einsum

MODEL_SIZES = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
    "10B": {"d_model": 4608, "d_ff": 12288, "num_layers": 50, "num_heads": 36},
}

@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = K.shape[-1]
    with nvtx.range("computing attention scores"):
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)
    if mask is not None:
        with nvtx.range("applying causal mask"):
            attention_scores = torch.where(mask, attention_scores, float("-inf"))
    with nvtx.range("computing softmax"):
        attention_weights = softmax(attention_scores, dim=-1)
    with nvtx.range("final matmul"):
        return einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")

def build_model(model_size, vocab_size, context_length, device):
    cfg = MODEL_SIZES[model_size]
    return BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
    ).to(device)

def forward_only(model, batch, targets, optimizer):
    with torch.inference_mode():
        with nvtx.range("forward"):
            model(batch)

def forward_backward(model, batch, targets, optimizer):
    optimizer.zero_grad(set_to_none=True)
    with nvtx.range("forward"):
        output = model(batch)
        loss = cross_entropy(output, targets)
    with nvtx.range("backward"):
        loss.backward()

def training_step(model, batch, targets, optimizer):
    optimizer.zero_grad(set_to_none=True)
    with nvtx.range("forward"):
        output = model(batch)
        loss = cross_entropy(output, targets)
    with nvtx.range("backward"):
        loss.backward()
    with nvtx.range("optimizer_step"):
        optimizer.step()

MODES = {
    "forward": forward_only,
    "forward_backward": forward_backward,
    "training_step": training_step,
}

def run_profile(args):
    if not torch.cuda.is_available():
        raise RuntimeError("This workload is intended for Nsight Systems on a CUDA GPU.")

    if args.annotate_attention:
        model_impl.scaled_dot_product_attention = annotated_scaled_dot_product_attention

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    model = build_model(args.model_size, args.vocab_size, args.context_length, device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    batch = torch.randint(args.vocab_size, (args.batch_size, args.context_length), device=device)
    targets = torch.randint(args.vocab_size, (args.batch_size, args.context_length), device=device)
    step_fn = MODES[args.mode]

    for _ in range(args.warmup_steps):
        step_fn(model, batch, targets, optimizer)
        torch.cuda.synchronize()

    elapsed = []
    for i in range(args.profile_steps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with nvtx.range("profile_step"):
            with nvtx.range(f"profile_step/{i}"):
                step_fn(model, batch, targets, optimizer)
        torch.cuda.synchronize()
        elapsed.append(time.perf_counter() - start)

    mean_s = sum(elapsed) / len(elapsed)
    print(
        {
            "model_size": args.model_size,
            "context_length": args.context_length,
            "mode": args.mode,
            "profile_steps": args.profile_steps,
            "mean_time": mean_s,
            "step_times": elapsed,
        }
    )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", choices=MODEL_SIZES.keys(), default="small")
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--mode", choices=MODES.keys(), default="forward")
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--profile-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--annotate-attention", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()

if __name__ == "__main__":
    run_profile(parse_args())
