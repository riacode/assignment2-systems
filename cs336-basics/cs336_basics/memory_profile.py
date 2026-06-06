from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import torch
from torch.utils.checkpoint import checkpoint
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

MODEL_SIZES = {
    "xl": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}

def _build_xl_model(vocab_size: int, context_length: int, device: str) -> BasicsTransformerLM:
    cfg = MODEL_SIZES["xl"]
    return BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
    ).to(device)

def _forward_with_checkpointing(model, batch, checkpoint_subset_size: int):
    if checkpoint_subset_size <= 0:
        return model(batch)

    x = model.token_embeddings(batch)
    layers = list(model.layers)

    for start in range(0, len(layers), checkpoint_subset_size):
        subset = layers[start : start + checkpoint_subset_size]

        def run_subset(x, subset=subset):
            for layer in subset:
                x = layer(x)
            return x

        x = checkpoint(run_subset, x, use_reentrant=False)

    x = model.ln_final(x)
    return model.lm_head(x)


def _run_step(model, batch, targets, optimizer, mode: str, use_bf16: bool, checkpoint_subset_size: int) -> None:
    context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()
    if mode == "forward":
        with torch.inference_mode():
            with context:
                _forward_with_checkpointing(model, batch, checkpoint_subset_size)
    elif mode == "training_step":
        if optimizer is None:
            raise ValueError("training_step mode requires an optimizer")
        optimizer.zero_grad(set_to_none=True)
        with context:
            output = _forward_with_checkpointing(model, batch, checkpoint_subset_size)
            loss = cross_entropy(output, targets)
        loss.backward()
        optimizer.step()

def run_memory_profile(*, context_length, mode, use_bf16, snapshot_path, batch_size=4, vocab_size=10000, num_warmup_steps=1, checkpoint_subset_size=0):
    device = "cuda"
    precision = "bf16" if use_bf16 else "fp32"
    snapshot_path = Path(snapshot_path)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "model_size": "xl",
        "context_length": context_length,
        "mode": mode,
        "precision": precision,
        "batch_size": batch_size,
        "vocab_size": vocab_size,
        "checkpoint_subset_size": checkpoint_subset_size,
        "status": "ok",
        "peak_memory_bytes": None,
        "peak_memory_gib": None,
        "snapshot_path": str(snapshot_path),
        "error": None,
    }

    try:
        model = _build_xl_model(vocab_size, context_length, device)
        batch = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
        targets = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
        optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8) if mode == "training_step" else None

        for _ in range(num_warmup_steps):
            _run_step(model, batch, targets, optimizer, mode, use_bf16, checkpoint_subset_size)
            torch.cuda.synchronize()
            model.zero_grad(set_to_none=True)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.memory._record_memory_history(max_entries=1000000)

        _run_step(model, batch, targets, optimizer, mode, use_bf16, checkpoint_subset_size)
        torch.cuda.synchronize()
        row["peak_memory"] = torch.cuda.max_memory_allocated() / (1024**3)
        torch.cuda.memory._dump_snapshot(str(snapshot_path))
        torch.cuda.memory._record_memory_history(enabled=None)
    except torch.OutOfMemoryError as exc:
        torch.cuda.memory._record_memory_history(enabled=None)
        row["error"] = str(exc)
        row["status"] = "oom"
    except Exception as exc:
        torch.cuda.memory._record_memory_history(enabled=None)
        row["error"] = repr(exc)
        row["status"] = "error"

    return row
