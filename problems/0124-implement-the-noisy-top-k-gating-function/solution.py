import torch
import torch.nn.functional as F

def noisy_topk_gating(
    X: torch.Tensor,
    W_g: torch.Tensor,
    W_noise: torch.Tensor,
    N: torch.Tensor,
    k: int
) -> torch.Tensor:
    """
    Args:
        X: Input data, shape (batch_size, features)
        W_g: Gating weight matrix, shape (features, num_experts)
        W_noise: Noise weight matrix, shape (features, num_experts)
        N: Noise samples, shape (batch_size, num_experts)
        k: Number of experts to keep per example
    Returns:
        Gating probabilities, shape (batch_size, num_experts)
    """
    # Your code here
    
    base = X @ W_g
    noise = X @ W_noise
    final = base + N * F.softplus(noise)
    values,indices = torch.topk(final,k)
    probs = F.softmax(values, dim=1)

    # same shape as final
    output = torch.zeros_like(final)
    output.scatter_(1, indices, probs)

    return output