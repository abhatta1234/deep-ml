import torch

def adam_optimizer(f, grad, x0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=10) -> torch.Tensor:
    """
    Implements Adam optimization algorithm using PyTorch's built-in optimizer.

    Args:
        f: The objective function to be optimized
        grad: A function that computes the gradient (unused; autograd is used instead)
        x0: Initial parameter values (torch.Tensor)
        learning_rate: The step size (default: 0.001)
        beta1: Exponential decay rate for the first moment estimates (default: 0.9)
        beta2: Exponential decay rate for the second moment estimates (default: 0.999)
        epsilon: A small constant for numerical stability (default: 1e-8)
        num_iterations: Number of iterations to run the optimizer (default: 10)

    Returns:
        torch.Tensor: Optimized parameters
    """

    # need leaf tensor to track gradients
    x = x0.detach().clone().requires_grad_(True)

    adam_optimizer=torch.optim.Adam([x], lr=learning_rate,betas=(beta1,beta2),eps=epsilon)

    for t in range(num_iterations):  
        # loss compute
        loss = f(x)
        #backward pass
        loss.backward()
        adam_optimizer.step()
        adam_optimizer.zero_grad()
    
    return x.detach()
       





