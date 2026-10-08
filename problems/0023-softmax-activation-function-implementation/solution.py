import torch
import torch.nn.functional as F

def softmax(scores: list[float]) -> list[float]:
    """
    Compute the softmax activation function using PyTorch's built-in API.
    Input:
      - scores: list of floats (logits)
    Returns:
      - list of floats representing the softmax probabilities.
    """
    # Your implementation here
    # Your code here
    scores_at = torch.as_tensor(scores)

    numerator = torch.exp(scores_at)
    
    denominator = torch.sum(torch.exp(scores_at))
    
    vals = numerator/denominator
    
    return vals.tolist()
