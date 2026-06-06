import torch
import torch.nn as nn

from cs336_basics.modal_utils import VOLUME_MOUNTS, app, build_image


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x

def inspect_autocast_dtypes():
    device = "cuda"
    model = ToyModel(in_features=8, out_features=4).to(device)
    x = torch.randn(2, 8, device=device)
    target = torch.randn(2, 4, device=device)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        fc1_output = model.fc1(x)
        relu_output = model.relu(fc1_output)
        layer_norm_output = model.ln(relu_output)
        logits = model.fc2(layer_norm_output)
        loss = nn.MSELoss()(logits, target)

    loss.backward()
    result = {
        "model parameters": str(next(model.parameters()).dtype),
        "fc1 output": str(fc1_output.dtype),
        "layer norm output": str(layer_norm_output.dtype),
        "logits": str(logits.dtype),
        "loss": str(loss.dtype),
        "gradients": str(model.fc1.weight.grad.dtype),
    }
    for name, dtype in result.items(): print(f"{name}: {dtype}")
    return result

@app.function(image=build_image(), volumes=VOLUME_MOUNTS, gpu="B200", retries=0)
def inspect_autocast_dtypes_remote():
    return inspect_autocast_dtypes()

@app.local_entrypoint()
def main():
    print(inspect_autocast_dtypes_remote.remote())
