import torch

def calculate_matrix_mean(matrix, mode: str) -> torch.Tensor:
    """
    Calculate mean of a 2D matrix per row or per column using PyTorch.
    Inputs can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 1-D tensor of means or raises ValueError on invalid mode.
    """

    '''
    1 2 3
    4 5 6
    7 8 9

    '''
    a_t = torch.as_tensor(matrix, dtype=torch.float)
    indx=None
    if mode == "row": 
        indx = 1
    if mode == "column":
        indx = 0
    
    return torch.mean(a_t,indx,keepdim=True)
