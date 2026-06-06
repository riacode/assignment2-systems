import torch
import torch.distributed as dist


class ShardedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls: type[torch.optim.Optimizer], **kwargs):
        self.all_param_groups = []
        self.local_param_groups = []
        super().__init__(params, kwargs)
        self.optimizer = optimizer_cls(self.local_param_groups, **kwargs)

    def add_param_group(self, param_group):
        # these parameters will be sharded across all the ranks
        params = list(param_group["params"])
        param_group = dict(param_group)
        param_group["params"] = params
        self.all_param_groups.append(param_group)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_params = [parameter for i, parameter in enumerate(params) if i % world_size == rank]
        if local_params:
            # each rank's optimizer instance will only handle a subset of the parameters
            local_param_group = dict(param_group)
            local_param_group["params"] = local_params
            self.local_param_groups.append(local_param_group)

    # added since test failed
    def zero_grad(self, set_to_none: bool = True):
        for group in self.all_param_groups:
            for parameter in group["params"]:
                parameter.grad = None if set_to_none else torch.zeros_like(parameter)

    def step(self, closure=None, **kwargs):
        loss = self.optimizer.step(closure=closure, **kwargs)
        world_size = dist.get_world_size()
        for group in self.all_param_groups:
            for i, parameter in enumerate(group["params"]):
                # broadcast updated parameters to other ranks
                dist.broadcast(parameter.data, src=i % world_size)
        return loss
