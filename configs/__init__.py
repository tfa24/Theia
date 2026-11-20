from .fine_tuning_config import (
    Theia_base_ft_384_config,
    Theia_base_ft_512_config,
    Theia_large_ft_384_config,
    Theia_large_ft_512_config,
    Theia_small_ft_384_config,
    Theia_small_ft_512_config,
    Theia_tiny_ft_384_config,
    Theia_tiny_ft_512_config,
)
from .model_config import (
    Theia_base_ce_config,
    Theia_base_config,
    Theia_large_config,
    Theia_small_ce_config,
    Theia_small_config,
    Theia_tiny_ce_config,
    Theia_tiny_config,
)

CONFIGS = {
    # Model configs
    "tiny": Theia_tiny_config,
    "small": Theia_small_config,
    "base": Theia_base_config,
    "large": Theia_large_config,
    "tiny_ce": Theia_tiny_ce_config,
    "small_ce": Theia_small_ce_config,
    "base_ce": Theia_base_ce_config,
    # Fine-tuning configs
    "tiny_ft_384": Theia_tiny_ft_384_config,
    "tiny_ft_512": Theia_tiny_ft_512_config,
    "small_ft_384": Theia_small_ft_384_config,
    "small_ft_512": Theia_small_ft_512_config,
    "base_ft_384": Theia_base_ft_384_config,
    "base_ft_512": Theia_base_ft_512_config,
    "large_ft_384": Theia_large_ft_384_config,
    "large_ft_512": Theia_large_ft_512_config,
}
