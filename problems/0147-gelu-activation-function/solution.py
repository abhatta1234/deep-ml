import torch

def GeLU(x: torch.Tensor) -> torch.Tensor:
    # Your code here
    erf_fn = 0.5*(1+torch.erf(x/torch.sqrt(torch.tensor(2))))
    return x * erf_fn
