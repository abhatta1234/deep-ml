import torch
from typing import List

def log_softmax(scores: List[float]) -> torch.Tensor:
    """
    Compute the log-softmax of a 1D list of scores using PyTorch.
    Args:
        scores: list of floats
    Returns:
        torch.Tensor of log-softmax values
    """
    scores = torch.as_tensor(scores)
    M = torch.max(scores)
    log_term = torch.log(torch.sum(torch.exp(scores-M)))
    return scores - M - log_term
    
