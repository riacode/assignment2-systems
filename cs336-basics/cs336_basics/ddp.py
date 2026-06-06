import torch
import torch.distributed as dist
import torch.nn as nn

class DDP(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        for parameter in self.module.parameters():
            dist.broadcast(parameter.data, src=0)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        parameters = [parameter for parameter in self.module.parameters() if parameter.grad is not None]
        gradients = [parameter.grad for parameter in parameters]
        flattened_gradients = torch._utils._flatten_dense_tensors(gradients)
        dist.all_reduce(flattened_gradients, op=dist.ReduceOp.SUM)
        flattened_gradients /= dist.get_world_size()
        unflattened_gradients = torch._utils._unflatten_dense_tensors(flattened_gradients, gradients)
        for i in range(len(parameters)):
            parameters[i].grad = unflattened_gradients[i]


class DDP2(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.handles = []
        self.gradients = []

        for parameter in self.module.parameters():
            dist.broadcast(parameter.data, src=0)
            if parameter.requires_grad:
                parameter.register_post_accumulate_grad_hook(self._start_gradient_synchronization)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def _start_gradient_synchronization(self, parameter):
        if parameter.grad is None:
            return
        self.gradients.append(parameter.grad)
        self.handles.append(dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM, async_op=True))

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        for gradient in self.gradients:
            gradient /= dist.get_world_size()
        self.handles.clear()
        self.gradients.clear()
