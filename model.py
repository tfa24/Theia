from collections import OrderedDict
from functools import partial

import lightning.pytorch as pl
import numpy as np
import torch
from timm.loss import SoftTargetCrossEntropy
from torch import Tensor, nn
from torch.optim import AdamW
from torchmetrics.classification import Accuracy
from torchvision.ops import StochasticDepth
from torchvision.ops.misc import Conv2dNormActivation, SqueezeExcitation

from aft_conv import AFTConv2d
from utils.grn import GRN
from utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from utils.ema import EMA


class MBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion_ratio: float,
        squeeze_ratio: float,
        stride: int,
        stochastic_depth: float,
    ) -> None:
        """
        Mobile Inverted Residual Bottleneck Block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            expansion_ratio (float): Expansion ratio in the bottleneck.
            squeeze_ratio (float): Squeeze ratio in the SE Layer.
            stride (int): Stride of the depthwise convolution.
            stochastic_depth (float): Drop probability for stochastic depth.
        """
        super().__init__()

        should_proj = stride != 1 or in_channels != out_channels
        if should_proj:
            proj = [
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=True)
            ]
            if stride == 2:
                proj = [nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)] + proj
            self.proj = nn.Sequential(*proj)
        else:
            self.proj = nn.Identity()

        mid_channels = int(out_channels * expansion_ratio)
        sqz_channels = int(out_channels * squeeze_ratio)

        if stochastic_depth:
            self.stochastic_depth = StochasticDepth(stochastic_depth, mode="row")
        else:
            self.stochastic_depth = nn.Identity()

        _layers = OrderedDict()
        _layers["pre_norm"] = nn.BatchNorm2d(in_channels, eps=1e-3, momentum=0.01)
        _layers["conv_a"] = Conv2dNormActivation(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_layer=partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01),
            activation_layer=nn.GELU,
            inplace=None,
        )
        _layers["conv_b"] = Conv2dNormActivation(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=mid_channels,
            norm_layer=partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01),
            activation_layer=nn.GELU,
            inplace=None,
        )
        _layers["squeeze_excitation"] = SqueezeExcitation(
            mid_channels, sqz_channels, activation=nn.SiLU
        )

        _layers["conv_c"] = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
        )

        self.layers = nn.Sequential(_layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor with expected layout of [B, C, H, W].
        Returns:
            Tensor: Output tensor with expected layout of [B, C, H // stride, W // stride].
        """
        res = self.proj(x)
        x = self.stochastic_depth(self.layers(x))
        return res + x


class AFTEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        aft_kernel_size: int,
        mlp_ratio: int,
        stochastic_depth: float,
    ) -> None:
        """
        AFTEncoder applies an GRN-AFTConv2d followed by an GRN-MLP.

        Args:
            in_channels (int): Number of input channels.
            aft_kernel_size (int): Kernel size for the AFTConv2d layer.
            mlp_ratio (int): Expansion factor for the MLP hidden dimension.
            stochastic_depth (float): Drop probability for stochastic depth.
        """
        super().__init__()

        self.attn_layer = nn.Sequential(
            nn.GroupNorm(1, in_channels),
            AFTConv2d(
                in_channels=in_channels,
                kernel_size=aft_kernel_size,
            ),
        )

        self.mlp_layer = nn.Sequential(
            nn.GroupNorm(1, in_channels),
            nn.Conv2d(in_channels, in_channels * mlp_ratio, kernel_size=1),
            nn.GELU(),
            GRN(in_channels * mlp_ratio),
            nn.Conv2d(in_channels * mlp_ratio, in_channels, kernel_size=1),
        )

        if stochastic_depth:
            self.stochastic_dropout = StochasticDepth(stochastic_depth, mode="row")
        else:
            self.stochastic_dropout = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape [B, C, H, W].

        Returns:
            Tensor: Output tensor of shape [B, C, H, W].
        """
        x = x + self.stochastic_dropout(self.attn_layer(x))
        x = x + self.stochastic_dropout(self.mlp_layer(x))

        return x


class TheiaLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        aft_kernel_size: int,
        mbconv_expansion_ratio: float,
        mbconv_squeeze_ratio: float,
        mlp_ratio: int,
        stochastic_depth: float,
    ) -> None:
        """
        TheiaLayer applies a MBConv block followed by two GRN-AFTEncoder.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride for the MBConv block.
            aft_kernel_size (int): Kernel size for the AFTConv2d layer.
            mbconv_expansion_ratio (float): Expansion ratio for the MBConv block.
            mbconv_squeeze_ratio (float): Squeeze ratio for the MBConv block.
            mlp_ratio (int): Expansion factor for the MLP hidden dimension.
            stochastic_depth (float): Drop probability for stochastic depth.
        """
        super().__init__()

        self.MBconv = MBConv(
            in_channels,
            out_channels,
            mbconv_expansion_ratio,
            mbconv_squeeze_ratio,
            stride,
            stochastic_depth,
        )
        self.aft = AFTEncoder(
            in_channels=out_channels,
            aft_kernel_size=aft_kernel_size,
            mlp_ratio=mlp_ratio,
            stochastic_depth=stochastic_depth,
        )
        self.aft2 = AFTEncoder(
            in_channels=out_channels,
            aft_kernel_size=aft_kernel_size,
            mlp_ratio=mlp_ratio,
            stochastic_depth=stochastic_depth,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape [B, C, H, W].

        Returns:
            Tensor: Output tensor of shape [B, C, H, W].
        """
        x = self.MBconv(x)
        x = self.aft(x)
        x = self.aft2(x)

        return x


class TheiaBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aft_kernel_size: int,
        mbconv_expansion_ratio: float,
        mbconv_squeeze_ratio: float,
        mlp_ratio: int,
        stochastic_depth: list[float],
    ) -> None:
        """
        TheiaBlock comprises multiple TheiaLayers. Each block downsamples the input spatially in the first layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            aft_kernel_size (int): Kernel size for the AFTConv2d layer.
            mbconv_expansion_ratio (float): Expansion ratio for the MBConv block.
            mbconv_squeeze_ratio (float): Squeeze ratio for the MBConv block.
            mlp_ratio (int): Expansion factor for the MLP hidden dimension.
            stochastic_depth (list[float]): List of drop probabilities for stochastic depth.
        """
        super().__init__()

        self.layers = nn.ModuleList()
        for idx, p in enumerate(stochastic_depth):
            stride = 2 if idx == 0 else 1
            self.layers.append(
                TheiaLayer(
                    in_channels=in_channels if idx == 0 else out_channels,
                    out_channels=out_channels,
                    stride=stride,
                    aft_kernel_size=aft_kernel_size,
                    mlp_ratio=mlp_ratio,
                    stochastic_depth=p,
                    mbconv_expansion_ratio=mbconv_expansion_ratio,
                    mbconv_squeeze_ratio=mbconv_squeeze_ratio,
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape [B, C, H, W].

        Returns:
            Tensor: Output tensor of shape [B, C, H, W].
        """
        for layer in self.layers:
            x = layer(x)
        return x


class Theia(pl.LightningModule):
    def __init__(
        # Model architecture parameters
        self,
        input_channels: int,
        stem_channels: int,
        block_channels: list[int],
        block_layers: list[int],
        aft_kernel_sizes: list[int],
        mbconv_expansion_ratio: float,
        mbconv_squeeze_ratio: float,
        mlp_ratio: int,
        stochastic_depth: float,
        num_classes: int,
        # Training parameters
        learning_rate: float,
        weight_decay: float,
        warmup_epochs: int,
        max_epochs: int,
        loss: str = "bce",
        # skip any dataloader parameters
        **kwargs,
    ) -> None:
        """
        Args:
            input_channels (int): Number of channels in the input image.
            stem_channels (int): Number of channels in the stem convolutions.
            block_channels (list[int]): List specifying the number of channels for each block.
            block_layers (list[int]): List specifying the number of layers in each block.
            aft_kernel_sizes (list[int]): List of kernel sizes for the AFTConv2d layers in each block.
            mbconv_expansion_ratio (float): Expansion ratio for the MBConv blocks.
            mbconv_squeeze_ratio (float): Squeeze ratio for the MBConv blocks.
            mlp_ratio (int): Expansion factor for the MLP hidden dimension.
            stochastic_depth (float): Maximum drop probability for stochastic depth.
            num_classes (int): Number of output classes.
            learning_rate (float): Optimizer learning rate.
            weight_decay (float): Weight decay (L2 regularization factor).
            warmup_epochs (int): Number of warmup epochs for the learning rate scheduler.
            max_epochs (int): Total number of training epochs.
            loss (str): Training loss, "ce" or "bce".
        """
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.loss = loss

        self.stem = nn.Sequential(
            Conv2dNormActivation(
                in_channels=input_channels,
                out_channels=stem_channels,
                kernel_size=3,
                stride=2,
                norm_layer=partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01),
                activation_layer=nn.GELU,
                bias=False,
                inplace=None,
            ),
            Conv2dNormActivation(
                in_channels=stem_channels,
                out_channels=stem_channels,
                kernel_size=3,
                stride=1,
                norm_layer=None,
                activation_layer=None,
                bias=True,
            ),
        )

        self.blocks = nn.ModuleList()
        in_channels_list = [stem_channels] + block_channels[:-1]
        out_channels_list = block_channels

        total_layers = sum(block_layers)
        sd_probs = np.linspace(0, stochastic_depth, total_layers).tolist()

        p_idx = 0
        for in_channel, out_channel, num_layers, aft_kernel_size in zip(
            in_channels_list, out_channels_list, block_layers, aft_kernel_sizes
        ):
            self.blocks.append(
                TheiaBlock(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    aft_kernel_size=aft_kernel_size,
                    mlp_ratio=mlp_ratio,
                    stochastic_depth=sd_probs[p_idx : p_idx + num_layers],
                    mbconv_expansion_ratio=mbconv_expansion_ratio,
                    mbconv_squeeze_ratio=mbconv_squeeze_ratio,
                )
            )
            p_idx += num_layers

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(block_channels[-1]),
            nn.Linear(block_channels[-1], block_channels[-1]),
            nn.Tanh(),
            nn.Linear(block_channels[-1], num_classes, bias=loss == "bce"),
        )

        self._init_weights()

        # Losses and metrics.
        if loss == "bce":
            self.train_loss = nn.BCEWithLogitsLoss(reduction="sum")
        else:
            self.train_loss = SoftTargetCrossEntropy()
        self.valid_loss = nn.CrossEntropyLoss()

        self.valid_acc_top_1 = Accuracy(
            task="multiclass", num_classes=num_classes, top_k=1
        )
        self.valid_acc_top_5 = Accuracy(
            task="multiclass", num_classes=num_classes, top_k=5
        )
        self.valid_acc_top_1_ema = Accuracy(
            task="multiclass", num_classes=num_classes, top_k=1
        )
        self.valid_acc_top_5_ema = Accuracy(
            task="multiclass", num_classes=num_classes, top_k=5
        )

        self._ema_cb: EMA | None = None

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        if self.loss == "bce":
            with torch.no_grad():
                self.classifier[-1].bias.fill_(-float(np.log(self.num_classes)))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape [B, C, H, W].

        Returns:
            Tensor: Output tensor of shape [B, num_classes].
        """

        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.classifier(x)

        return x

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        sample, target = batch

        pred = self(sample)
        loss = self.train_loss(pred, target)

        if self.loss == "bce":
            loss = loss / target.shape[0]

        self.log(
            "Training Loss",
            loss,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )

        return loss

    def on_validation_epoch_start(self):
        if self._ema_cb is None:
            for cb in self.trainer.callbacks:
                if isinstance(cb, EMA):
                    self._ema_cb = cb
                    break

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        sample, target = batch

        pred = self(sample)
        loss = self.valid_loss(pred, target)

        self.log(
            "Validation Accuracy Top 1",
            self.valid_acc_top_1(pred, target),
            on_epoch=True,
        )
        self.log(
            "Validation Accuracy Top 5",
            self.valid_acc_top_5(pred, target),
            on_epoch=True,
        )
        self.log(
            "Validation Loss",
            loss,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )

        if self._ema_cb:
            with self._ema_cb.use_ema_weights(self):
                pred_ema = self(sample)
                loss_ema = self.valid_loss(pred_ema, target)

            self.log(
                "EMA Validation Accuracy Top 1",
                self.valid_acc_top_1_ema(pred_ema, target),
                on_epoch=True,
            )
            self.log(
                "EMA Validation Accuracy Top 5",
                self.valid_acc_top_5_ema(pred_ema, target),
                on_epoch=True,
            )
            self.log(
                "EMA Validation Loss",
                loss_ema,
                on_epoch=True,
                on_step=False,
                sync_dist=True,
            )

        return loss

    def configure_optimizers(self) -> dict:
        wd, no_wd = [], []

        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                no_wd.append(m.weight)
                if m.bias is not None:
                    no_wd.append(m.bias)
            else:
                for name, param in m.named_parameters(recurse=False):
                    if not param.requires_grad:
                        continue
                    elif "no_wd" in name or "bias" in name:
                        no_wd.append(param)
                    else:
                        wd.append(param)

        param_groups = [
            {"params": wd, "weight_decay": self.weight_decay},
            {"params": no_wd, "weight_decay": 0.0},
        ]

        optimizer = AdamW(param_groups, lr=self.learning_rate)

        lr_scheduler_config = {
            "scheduler": LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=self.warmup_epochs,
                max_epochs=self.max_epochs,
                warmup_start_lr=1e-5,
                eta_min=1e-5,
            ),
            "interval": "epoch",
            "frequency": 1,
            "name": "Learning Rate",
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
