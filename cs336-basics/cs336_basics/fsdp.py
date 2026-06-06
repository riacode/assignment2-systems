import torch
import torch.distributed as dist
import torch.nn as nn
from cs336_basics.model import Embedding, Linear


class FSDP(nn.Module):
    def __init__(self, module: nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()
        self.module = module
        self.compute_dtype = compute_dtype
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.params = {}
        for parameter in self.module.parameters():
            dist.broadcast(parameter.data, src=0)
        # hook into or wrap any Linear or Embedding layer
        for name, mod in self.module.named_modules():
            if isinstance(mod, (Linear, Embedding)):
                parameter = mod.weight
                self.params[f"{name}.weight"] = parameter
                # each GPU stores only its own slice of every weight tensor
                shards = parameter.data.chunk(self.world_size, dim=0)
                parameter.data = shards[self.rank].contiguous()

    def forward(self, *args, **kwargs):
        for parameter in self.params.values():
            parameter.shard = parameter.data
            gathered = [torch.empty_like(parameter.data) for _ in range(self.world_size)]
            # all-gather the weights
            dist.all_gather(gathered, parameter.data)
            parameter.data = torch.cat(gathered, dim=0)
            # when compute_dtype is provided cast the weights to that dtype
            if self.compute_dtype is not None:
                parameter.data = parameter.data.to(self.compute_dtype)
        # calls module's forward()
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        for parameter in self.params.values():
            # master weights and optimizer updates in FP32
            gradient = parameter.grad.to(torch.float32)
            shard_grad = torch.empty_like(parameter.shard, dtype=torch.float32)
            if dist.get_backend() == "gloo":
                dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
                shard_grad = gradient.chunk(self.world_size, dim=0)[self.rank].contiguous()
            else:
                dist.reduce_scatter_tensor(shard_grad, gradient.contiguous(), op=dist.ReduceOp.SUM)
            shard_grad /= self.world_size
            # free gathered weights after use
            parameter.data = parameter.shard
            parameter.grad = shard_grad.to(parameter.dtype).contiguous()
        for name, parameter in self.module.named_parameters():
            if name in self.params or parameter.grad is None:
                continue
            # average their gradients
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.AVG)


def gather_full_params(model):
    full_params = {}
    for name, parameter in model.module.named_parameters():
        if name in model.params:
            gathered = [torch.empty_like(parameter.data) for _ in range(model.world_size)]
            dist.all_gather(gathered, parameter.data)
            full_params[name] = torch.cat(gathered, dim=0)
        else:
            full_params[name] = parameter.data.clone()
    return full_params
