import torch
import torch.nn.functional as F

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> tuple[list[float], float]:
    """
    Simulates a single neuron with sigmoid activation for binary classification.
    
    Args:
        features: List of feature vectors (each a list of floats)
        labels: List of true binary labels
        weights: Neuron weights (one per feature)
        bias: Neuron bias term
    
    Returns:
        Tuple of (predicted probabilities rounded to 4 decimal places, MSE rounded to 4 decimal places)
    """
    # Your code here using PyTorch built-ins:
    # - torch.matmul() for linear combination
    # - torch.sigmoid() for activation
    # - torch.nn.functional.mse_loss() for MSE

    # step 1: initialization
    features = torch.as_tensor(features).float()
    labels = torch.as_tensor(labels).float()
    weights = torch.as_tensor(weights).float()
    bias = torch.as_tensor(bias).float()

    # step 2: forward pass
    output = torch.matmul(features, weights) + bias
    probs = torch.sigmoid(output)

    # loss compute
    loss = F.mse_loss(probs,labels)
    return_probs = [round(x,4) for x in probs.tolist()]

    return return_probs, round(loss.item(),4)

