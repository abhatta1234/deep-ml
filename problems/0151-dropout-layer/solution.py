import torch

class DropoutLayer:
    def __init__(self, p: float):
        """Initialize the dropout layer.
        
        Attributes to set:
            self.p: the dropout rate
            self.mask: stores the dropout mask (initially None)
        """
        self.p = p 
        self.mask = None

    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Forward pass of the dropout layer.
        
        Generate a new mask on each training forward pass and store it in self.mask.
        """
        if training:
            self.mask = torch.where(torch.rand_like(x) > self.p, 1, 0) 
            return (x*self.mask) / (1 - self.p)
        else:
            self.mask=None
            return x

    def backward(self, grad: torch.Tensor) -> torch.Tensor:
        """Backward pass of the dropout layer.
        
        Use the stored self.mask from the most recent forward pass.
        """
        if self.mask is None:
            return grad
        else:
            return (grad*self.mask) / (1 - self.p)
    