import torch.nn as nn


def count_params(model: nn.Module):
    return {
        "params-total": sum(p.numel() for p in model.parameters()),
        "params-trainable": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "params-not-trainable": sum(
            p.numel() for p in model.parameters() if not p.requires_grad
        ),
    }
