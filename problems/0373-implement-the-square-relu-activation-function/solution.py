import torch

def square_relu(x: torch.Tensor) -> dict:
    """
    Apply the Square ReLU activation function and compute its derivative.
    
    Args:
        x: Input torch.Tensor of any shape
    
    Returns:
        Dictionary with 'output' and 'derivative' as torch.Tensors
    """
    return {"output":torch.where(x>0,x**2,0),
    "derivative":torch.where(x>0,2*x,0)}