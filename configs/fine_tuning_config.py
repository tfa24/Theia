"""
Resolution fine-tuning configurations for Theia models.

Default configuration uses a batch size of 512 with a learning rate of 3e-3, ema decay of 0.9999.
    lr = 3e-3 * (effective_batch_size / 512)
    ema = 0.9999 ** (effective_batch_size / 512)
ema starts immediately.

Please adjust the learning rate and batch size based on your hardware capabilities.

The following configs are used for training on TPUv4-64 at fp32.
"""

from dataclasses import dataclass
from .model_config import *


@dataclass
class Theia_tiny_ft_config(Theia_tiny_config):
    # Model architecture parameters
    num_classes: int = 1000
    # Training parameters
    learning_rate: float = 1e-5
    weight_decay: float = 0.1
    warmup_epochs: int = 5
    max_epochs: int = 40
    ema: float = 0.9999
    ema_steps: int = 1
    ema_start_step: int = 0
    # Dataloader parameters
    train_batch_size: int = 16
    eval_batch_size: int = 16


@dataclass
class Theia_tiny_ft_384_config(Theia_tiny_config):
    # Model architecture parameters
    num_classes: int = 1000
    # Training parameters
    learning_rate: float = 1e-5
    weight_decay: float = 0.1
    warmup_epochs: int = 5
    max_epochs: int = 40
    ema: float = 0.9999
    ema_steps: int = 1
    ema_start_step: int = 0
    # Dataloader parameters
    train_batch_size: int = 16
    train_image_size: int = 384
    eval_batch_size: int = 16
    eval_image_size: int = 384


@dataclass
class Theia_tiny_ft_512_config(Theia_tiny_config):
    # Model architecture parameters
    num_classes: int = 1000
    # Training parameters
    learning_rate: float = 1e-5
    weight_decay: float = 0.1
    warmup_epochs: int = 5
    max_epochs: int = 40
    ema: float = 0.9999
    ema_steps: int = 1
    ema_start_step: int = 0
    # Dataloader parameters
    train_batch_size: int = 16
    train_image_size: int = 512
    eval_batch_size: int = 16
    eval_image_size: int = 512


@dataclass
class Theia_small_ft_config(Theia_small_config):
    # Model architecture parameters
    num_classes: int = 1000
    # Training parameters
    learning_rate: float = 1e-5
    weight_decay: float = 0.1
    warmup_epochs: int = 5
    max_epochs: int = 40
    ema: float = 0.9999
    ema_steps: int = 1
    ema_start_step: int = 0
    # Dataloader parameters
    train_batch_size: int = 16
    eval_batch_size: int = 16


@dataclass
class Theia_small_ft_384_config(Theia_small_config):
    # Model architecture parameters
    num_classes: int = 1000
    # Training parameters
    learning_rate: float = 1e-5
    weight_decay: float = 0.1
    warmup_epochs: int = 5
    max_epochs: int = 40
    ema: float = 0.9999
    ema_steps: int = 1
    ema_start_step: int = 0
    # Dataloader parameters
    train_batch_size: int = 16
    train_image_size: int = 384
    eval_batch_size: int = 16
    eval_image_size: int = 384


@dataclass
class Theia_small_ft_512_config(Theia_small_config):
    # Model architecture parameters
    num_classes: int = 1000
    # Training parameters
    learning_rate: float = 1e-5
    weight_decay: float = 0.1
    warmup_epochs: int = 5
    max_epochs: int = 40
    ema: float = 0.9999
    ema_steps: int = 1
    ema_start_step: int = 0
    # Dataloader parameters
    train_batch_size: int = 16
    train_image_size: int = 512
    eval_batch_size: int = 16
    eval_image_size: int = 512


@dataclass
class Theia_base_ft_config(Theia_base_config):
    # Model architecture parameters
    num_classes: int = 1000
    # Training parameters
    learning_rate: float = 1e-5
    weight_decay: float = 0.1
    warmup_epochs: int = 5
    max_epochs: int = 40
    ema: float = 0.9999
    ema_steps: int = 1
    ema_start_step: int = 0
    # Dataloader parameters
    train_batch_size: int = 16
    eval_batch_size: int = 16


@dataclass
class Theia_base_ft_384_config(Theia_base_config):
    # Model architecture parameters
    num_classes: int = 1000
    # Training parameters
    learning_rate: float = 1e-5
    weight_decay: float = 0.1
    warmup_epochs: int = 5
    max_epochs: int = 40
    ema: float = 0.9999
    ema_steps: int = 1
    ema_start_step: int = 0
    # Dataloader parameters
    train_batch_size: int = 16
    train_image_size: int = 384
    eval_batch_size: int = 16
    eval_image_size: int = 384


@dataclass
class Theia_base_ft_512_config(Theia_base_config):
    # Model architecture parameters
    num_classes: int = 1000
    # Training parameters
    learning_rate: float = 1e-5
    weight_decay: float = 0.1
    warmup_epochs: int = 5
    max_epochs: int = 40
    ema: float = 0.9999
    ema_steps: int = 1
    ema_start_step: int = 0
    # Dataloader parameters
    train_batch_size: int = 16
    train_image_size: int = 512
    eval_batch_size: int = 16
    eval_image_size: int = 512


@dataclass
class Theia_large_ft_config(Theia_large_config):
    # Model architecture parameters
    num_classes: int = 1000
    # Training parameters
    learning_rate: float = 1e-5
    weight_decay: float = 0.1
    warmup_epochs: int = 5
    max_epochs: int = 40
    ema: float = 0.9999
    ema_steps: int = 1
    ema_start_step: int = 0
    # Dataloader parameters
    train_batch_size: int = 16
    eval_batch_size: int = 16


@dataclass
class Theia_large_ft_384_config(Theia_large_config):
    # Model architecture parameters
    num_classes: int = 1000
    # Training parameters
    learning_rate: float = 1e-5
    weight_decay: float = 0.1
    warmup_epochs: int = 5
    max_epochs: int = 40
    ema: float = 0.9999
    ema_steps: int = 1
    ema_start_step: int = 0
    # Dataloader parameters
    train_batch_size: int = 16
    train_image_size: int = 384
    eval_batch_size: int = 16
    eval_image_size: int = 384


@dataclass
class Theia_large_ft_512_config(Theia_large_config):
    # Model architecture parameters
    num_classes: int = 1000
    # Training parameters
    learning_rate: float = 1e-5
    weight_decay: float = 0.1
    warmup_epochs: int = 5
    max_epochs: int = 40
    ema: float = 0.9999
    ema_steps: int = 1
    ema_start_step: int = 0
    # Dataloader parameters
    train_batch_size: int = 16
    train_image_size: int = 512
    eval_batch_size: int = 16
    eval_image_size: int = 512
