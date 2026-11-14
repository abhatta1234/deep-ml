import torch

def SwiGLU(x: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: torch.Tensor of shape (batch_size, 2d)
    Returns:
        torch.Tensor of shape (batch_size, d)
    """
    # Your code here
    x1,x2 = torch.chunk(x,chunks=2,dim=-1)
    return x1* (x2*torch.sigmoid(x2))
