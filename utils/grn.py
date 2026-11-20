from torch import nn, Tensor
import torch


class GRN(nn.Module):
    def __init__(self, in_channels: int) -> None:
        """
        Global Response Normalization layer

        Args:
            dim (int): Number of channels in the input tensor.
        """
        super().__init__()

        self.gamma_no_wd = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
        self.beta_no_wd = nn.Parameter(torch.zeros(1, in_channels, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor with expected layout of [B, C, H, W].
        Returns:
            Tensor: Output tensor with expected layout of [B, C, H, W].
        """
        Gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma_no_wd * (x * Nx) + self.beta_no_wd + x
