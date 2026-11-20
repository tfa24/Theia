"""
Pre-training configs for Theia models.

Default configuration uses a batch size of 4096 with a learning rate of 3e-3, ema decay of 0.9999.
    lr = 3e-3 * (effective_batch_size / 4096)
    ema = 0.9999 ** (effective_batch_size / 4096)
ema starts after warmup epochs.

Please adjust the learning rate and batch size based on your hardware capabilities.

The following configs are used for training on TPUv4-64 at fp32.
"""

from dataclasses import dataclass, field


@dataclass
class Theia_tiny_config:
    # Model architecture parameters
    input_channels: int = 3
    stem_channels: int = 64
    block_channels: list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    block_layers: list[int] = field(default_factory=lambda: [2, 2, 5, 2])
    aft_kernel_sizes: list[int] = field(default_factory=lambda: [7, 7, 5, 3])
    mbconv_expansion_ratio: float = 4
    mbconv_squeeze_ratio: float = 0.25
    mlp_ratio: float = 4
    stochastic_depth: float = 0.2
    num_classes: int = 1000
    # Training parameters
    learning_rate: float = 3e-3
    weight_decay: float = 0.075
    warmup_epochs: int = 32
    max_epochs: int = 400
    ema: float = 0.9999
    ema_steps: int = 1
    ema_start_step: int = 9984
    # Dataloader parameters
    train_batch_size: int = 128
    train_image_size: int = 224
    color_jitter: float = 0.3
    mixup_alpha: float = 0.8
    cutmix_alpha: float = 1.0
    label_smoothing: float = 0.0
    eval_batch_size: int = 128
    eval_image_size: int = 224
    eval_ratio: float = 1.0


@dataclass
class Theia_small_config:
    # Model architecture parameters
    input_channels: int = 3
    stem_channels: int = 64
    block_channels: list[int] = field(default_factory=lambda: [96, 192, 384, 768])
    block_layers: list[int] = field(default_factory=lambda: [2, 2, 5, 2])
    aft_kernel_sizes: list[int] = field(default_factory=lambda: [7, 7, 5, 3])
    mbconv_expansion_ratio: float = 4
    mbconv_squeeze_ratio: float = 0.25
    mlp_ratio: float = 4
    stochastic_depth: float = 0.3
    num_classes: int = 1000
    # Training parameters
    learning_rate: float = 3e-3
    weight_decay: float = 0.075
    warmup_epochs: int = 32
    max_epochs: int = 400
    ema: float = 0.9999
    ema_steps: int = 1
    ema_start_step: int = 9984
    # Dataloader parameters
    train_batch_size: int = 128
    train_image_size: int = 224
    color_jitter: float = 0.3
    mixup_alpha: float = 0.8
    cutmix_alpha: float = 1.0
    label_smoothing: float = 0.0
    eval_batch_size: int = 128
    eval_image_size: int = 224
    eval_ratio: float = 1.0


@dataclass
class Theia_base_config:
    # Model architecture parameters
    input_channels: int = 3
    stem_channels: int = 64
    block_channels: list[int] = field(default_factory=lambda: [96, 192, 384, 768])
    block_layers: list[int] = field(default_factory=lambda: [2, 6, 14, 2])
    aft_kernel_sizes: list[int] = field(default_factory=lambda: [7, 7, 5, 3])
    mbconv_expansion_ratio: float = 4
    mbconv_squeeze_ratio: float = 0.25
    mlp_ratio: float = 4
    stochastic_depth: float = 0.4
    num_classes: int = 1000
    # Training parameters
    learning_rate: float = 3e-3
    weight_decay: float = 0.075
    warmup_epochs: int = 32
    max_epochs: int = 400
    ema: float = 0.9999
    ema_steps: int = 1
    ema_start_step: int = 9984
    # Dataloader parameters
    train_batch_size: int = 128
    train_image_size: int = 224
    color_jitter: float = 0.3
    mixup_alpha: float = 0.8
    cutmix_alpha: float = 1.0
    label_smoothing: float = 0.0
    eval_batch_size: int = 128
    eval_image_size: int = 224
    eval_ratio: float = 1.0


@dataclass
class Theia_large_config:
    # Model architecture parameters
    input_channels: int = 3
    stem_channels: int = 128
    block_channels: list[int] = field(default_factory=lambda: [128, 256, 512, 1024])
    block_layers: list[int] = field(default_factory=lambda: [2, 6, 14, 2])
    aft_kernel_sizes: list[int] = field(default_factory=lambda: [7, 7, 5, 3])
    mbconv_expansion_ratio: float = 4
    mbconv_squeeze_ratio: float = 0.25
    mlp_ratio: float = 4
    stochastic_depth: float = 0.6
    num_classes: int = 1000
    # Training parameters
    learning_rate: float = 1.875e-3
    weight_decay: float = 0.075
    warmup_epochs: int = 32
    max_epochs: int = 400
    ema: float = 0.9999375
    ema_steps: int = 1
    ema_start_step: int = 16000
    # Dataloader parameters
    train_batch_size: int = 80
    train_image_size: int = 224
    color_jitter: float = 0.3
    mixup_alpha: float = 0.8
    cutmix_alpha: float = 1.0
    label_smoothing: float = 0.0
    eval_batch_size: int = 80
    eval_image_size: int = 224
    eval_ratio: float = 1.0


@dataclass
class Theia_tiny_ce_config(Theia_tiny_config):
    loss: str = "ce"
    weight_decay: float = 0.05
    label_smoothing: float = 0.1


@dataclass
class Theia_small_ce_config(Theia_small_config):
    loss: str = "ce"
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    stochastic_depth: float = 0.4


@dataclass
class Theia_base_ce_config(Theia_base_config):
    loss: str = "ce"
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    stem_channels: int = 96
