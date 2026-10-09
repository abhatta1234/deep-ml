import torch
import torch.nn.functional as F
torch.backends.nnpack.set_flags(False)
def simple_conv2d(input_matrix: torch.Tensor, kernel: torch.Tensor, padding: int, stride: int) -> torch.Tensor:
    """
    Perform a 2D convolution on a single-channel input using PyTorch's built-in conv2d.
    input_matrix: 2D tensor (H, W)
    kernel: 2D tensor (kH, kW)
    padding: int, zero-padding on all sides
    stride: int, stride of the convolution
    """
    # Hint: conv2d expects input of shape (N, C, H, W) and weight of shape (out_channels, in_channels, kH, kW)
    
    # find the output matrix shape

    input_matrix_1= input_matrix.unsqueeze(0).unsqueeze(0)
    kernel_1 = kernel.unsqueeze(0).unsqueeze(0)
    #raise ValueError(f"{input_matrix.size(),kernel.size(),input_matrix_1.size(),kernel_1.size()}")

    return F.conv2d(input_matrix_1, kernel_1, padding=padding,stride=stride).squeeze().squeeze()
