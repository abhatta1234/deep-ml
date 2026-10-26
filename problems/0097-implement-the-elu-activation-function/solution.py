import torch
import torch.nn.functional as F

def elu(x: float, alpha: float = 1.0) -> float:
    """
    Compute the ELU activation function using PyTorch.

    Args:
        x (float): Input value
        alpha (float): ELU parameter for negative values (default: 1.0)

    Returns:
        float: ELU activation value rounded to 4 decimal places
    """
    # Your code here
    x = torch.as_tensor(x)
    alpha = torch.as_tensor(alpha)
    return torch.where(x>0,x,alpha*(torch.exp(x)-1))