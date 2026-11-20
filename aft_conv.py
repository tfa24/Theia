import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from utils.grn import GRN


class AFTConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        hidden_channels: int = None,
        out_channels: int = None,
    ) -> None:
        """
        Attention Free Transformer with Conv2d local information capture.

        Args:
            in_channels (int): Number of input channels.
            kernel_size (int): Kernel size for the convolution.
            hidden_channels (int): Number of hidden channels. If None, defaults to in_channels.
            out_channels (int): Number of output channels. If None, defaults to in_channels.
        """
        super().__init__()

        self.hidden_channels = hidden_channels if hidden_channels else in_channels
        self.out_channels = out_channels if out_channels else in_channels

        self.kernel_size = kernel_size

        self.to_qkv = nn.Conv2d(
            in_channels,
            3 * self.hidden_channels,
            kernel_size=1,
        )

        self.q_norm = nn.Sequential(nn.Sigmoid(), GRN(self.hidden_channels))
        self.k_norm = GRN(self.hidden_channels)
        self.v_norm = GRN(self.hidden_channels)

        self.out_proj = nn.Conv2d(
            self.hidden_channels, self.out_channels, kernel_size=1
        )

        self.conv_weight = nn.Parameter(
            torch.empty(self.hidden_channels, 1, kernel_size, kernel_size)
        )
        self.gamma_no_wd = nn.Parameter(torch.empty(self.hidden_channels, 1, 1, 1))
        self.beta_no_wd = nn.Parameter(torch.empty(self.hidden_channels, 1, 1, 1))

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.conv_weight, std=0.02)
        nn.init.zeros_(self.gamma_no_wd)
        nn.init.zeros_(self.beta_no_wd)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor with expected layout of [B, C, H, W].
        Returns:
            Tensor: Output tensor with expected layout of [B, C, H, W].
        """
        QKV = self.to_qkv(x)
        Q, K, V = torch.split(
            QKV,
            [self.hidden_channels, self.hidden_channels, self.hidden_channels],
            dim=1,
        )
        Q_sig = self.q_norm(Q)
        K = self.k_norm(K)
        V = self.v_norm(V)

        # Conv2d kernel
        mean = self.conv_weight.mean(dim=(1, 2, 3), keepdim=True)
        # std = self.conv_weight.std(
        #     dim=(1, 2, 3), unbiased=False, keepdim=True
        # )  # TPU hates this
        std = torch.sqrt(
            self.conv_weight.var(dim=(1, 2, 3), unbiased=False, keepdim=True)
        )  # but loves this???????
        conv_weight = (
            self.gamma_no_wd * ((self.conv_weight - mean) / std) + self.beta_no_wd
        )
        conv_weight = torch.exp(conv_weight) - 1

        # Denominator
        exp_K = torch.exp(K - K.amax(dim=(2, 3), keepdim=True))
        denom_global = torch.sum(exp_K, dim=(2, 3), keepdim=True)

        denom_local = F.conv2d(
            exp_K,
            weight=conv_weight,
            stride=1,
            padding=self.kernel_size // 2,
            groups=self.hidden_channels,
        )

        # Numerator
        exp_K_V = exp_K * V
        numer_global = torch.sum(exp_K_V, dim=(2, 3), keepdim=True)

        numer_local = F.conv2d(
            exp_K_V,
            weight=conv_weight,
            stride=1,
            padding=self.kernel_size // 2,
            groups=self.hidden_channels,
        )

        # AFT
        Y = Q_sig * ((numer_local + numer_global) / (denom_local + denom_global))
        Y = self.out_proj(Y)

        return Y
