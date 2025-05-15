import torch
from torch import nn, SymInt
from torch.nn import functional as F
from typing import Callable, Sequence, Union

class AverageConvnd(nn.Module):
    kernel: torch.Tensor
    conv_fn: Callable[..., torch.Tensor]

    def __init__(self, size: Sequence[Union[int, SymInt]]):
        super().__init__()
        self.size = size

        ndim = len(size)
        if ndim not in [1, 2, 3]:
            raise ValueError("length of `size` must be 1, 2 or 3.")
        
        self.register_buffer("kernel", self._make_average_kernel(size))
        self.conv_fn = getattr(F, f"conv{ndim}d")
    def _make_average_kernel(self, size: Sequence[Union[int, SymInt]]) -> torch.Tensor:
        kernel = torch.ones(size) / torch.as_tensor(size).prod()
        kernel = kernel.view(1,1,*kernel.shape)
        return kernel
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        in_channels = input.size(1)
        kernel = self.kernel.expand(in_channels, 1, *self.kernel.shape)
        input = self.conv_fn(input, kernel, padding = self.size // 2, groups=in_channels)
        return input

class GaussianConvnd(nn.Module):
    kernel: torch.Tensor
    conv_fn: Callable[..., torch.Tensor]

    def __init__(self, size: Sequence[Union[int, SymInt]], *, sigma: float = 1.5):
        super().__init__()
        self.size = size
        self.sigma = sigma

        ndim = len(size)
        if ndim not in [1, 2, 3]:
            raise ValueError("length of `size` must be 1, 2 or 3.")
        
        self.register_buffer("kernel", self._make_gaussian_kernel(size, sigma = sigma))
        self.conv_fn = getattr(F, f"conv{ndim}d")
        
    def _make_gaussian_kernel(self, size: Sequence[Union[int, SymInt]], *,  sigma: float = 1.5) -> torch.Tensor:
        coords = [torch.arange(size_dim, dtype=torch.float32) - (size_dim - 1) / 2.0 for size_dim in size]
        grids = torch.meshgrid(*coords, indexing="ij")
        kernel = torch.exp(-torch.stack(grids).pow(2).sum(0) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1,1,*kernel.shape)
        return kernel
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        in_channels = input.size(1)
        kernel = self.kernel.expand(in_channels, 1, *self.kernel.shape)
        input = self.conv_fn(input, kernel, padding = self.size // 2, groups=in_channels)
        return input
