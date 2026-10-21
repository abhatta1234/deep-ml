import torch

def hard_sigmoid(x: float) -> float:
    """
    Implements the Hard Sigmoid activation function using PyTorch.
    Uses the Keras convention: 0.2*x + 0.5, clamped to [0, 1].

    Args:
        x (float): Input value

    Returns:
        float: The Hard Sigmoid of the input
    """
    # Your code here
    # the whole range of output is [0,1] 
    # so return th function value in linear range else 0 or 1
    x = torch.tensor(x,dtype=torch.float32)
    return torch.clamp(0.2*x+0.5,0,1).item()