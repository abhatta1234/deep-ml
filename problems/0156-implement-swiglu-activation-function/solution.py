import torch

def SwiGLU(x: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: torch.Tensor of shape (batch_size, 2d)
    Returns:
        torch.Tensor of shape (batch_size, d)
    """
    # Your code here
    n = x.shape[-1]//2
    x1 = x[:,:n]
    x2 = x[:, n:]
    return x1* (x2*torch.sigmoid(x2))
