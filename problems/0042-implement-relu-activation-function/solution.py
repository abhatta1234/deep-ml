import torch

def relu(z: float) -> torch.Tensor:
    """
    Implements the ReLU activation function using PyTorch.
    
    Args:
        z: A float input value.
    
    Returns:
        A torch.Tensor with ReLU applied (max(0, z)).
    """
    # Your code here
    z_t = torch.as_tensor(z)
    # if z_t>0:
    #     return z_t
    # else:
    #     return torch.tensor(0)

    return torch.clamp(z_t,min=0)