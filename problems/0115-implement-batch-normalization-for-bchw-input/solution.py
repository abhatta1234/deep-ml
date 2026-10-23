import torch

def batch_normalization(
    X: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    epsilon: float = 1e-5
) -> torch.Tensor:

    mean = X.mean(dim=(0, 2, 3), keepdim=True)
    var = X.var(dim=(0, 2, 3), keepdim=True, correction=0)

    norm = (X - mean) / torch.sqrt(var + epsilon)

    gamma = gamma.reshape(1, -1, 1, 1)
    beta = beta.reshape(1, -1, 1, 1)

    return gamma * norm + beta