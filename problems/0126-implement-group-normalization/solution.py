import torch

def group_normalization(X: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, num_groups: int, epsilon: float = 1e-5) -> torch.Tensor:
    """
    Perform Group Normalization on a 4D input tensor.
    
    Args:
        X: torch tensor of shape (B, C, H, W), input data
        gamma: torch tensor of shape (1, C, 1, 1), scale parameter
        beta: torch tensor of shape (1, C, 1, 1), shift parameter
        num_groups: number of groups for normalization
        epsilon: small constant to avoid division by zero
    
    Returns:
        norm_X: torch tensor of shape (B, C, H, W), normalized output
    """
    B,C,H,W = X.shape
    G = C//num_groups
    X = torch.reshape(X,(B,num_groups,G,H,W))
    mean = torch.mean(X,dim=(2,3,4),keepdim=True)
    var = torch.var(X,dim=(2,3,4),correction=0,keepdim=True)
    X = (X-mean)/torch.sqrt(var+epsilon)
    X = torch.reshape(X,(B,num_groups*G,H,W))
    X = gamma*X+beta
    return X