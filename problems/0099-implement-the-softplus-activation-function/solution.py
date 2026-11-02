import torch
import torch.nn.functional as F

def softplus(x: float) -> torch.Tensor:
    """
    Compute the softplus activation function using PyTorch.

    Args:
        x: Input value

    Returns:
        The softplus value: log(1 + e^x) as a torch.Tensor
    """
    # Your code here
    x = torch.as_tensor(x)
    return max(x,0) + torch.log(1+torch.exp(-torch.abs(x)))