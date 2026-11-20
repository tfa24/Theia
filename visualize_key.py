import argparse
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

from aft_conv import AFTConv2d
from model import Theia


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to sample image",
    )
    parser.add_argument(
        "--res",
        type=int,
        default=224,
        help="Input resolution of the model",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="key_visualization/",
        help="Output directory for visualizations",
    )
    return parser.parse_args()


def extract_keys(model: Theia, sample_input: torch.Tensor) -> dict[str, torch.Tensor]:
    key_maps = {}

    def hook_fn(module, _input, _output, layer_name):
        qkv = module.to_qkv(_input[0])
        _, key, _ = torch.split(qkv, [module.hidden_channels] * 3, dim=1)
        key_maps[layer_name] = key.detach()

    for name, module in model.named_modules():
        if isinstance(module, AFTConv2d):
            module.register_forward_hook(partial(hook_fn, layer_name=name))

    model(sample_input)

    return key_maps


def visualize_keys(key_maps: dict[str, torch.Tensor], output_dir: str) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_min = min(k.min().item() for k in key_maps.values())
    global_max = max(k.max().item() for k in key_maps.values())

    for layer_name, key in sorted(key_maps.items()):
        layer_name = layer_name.replace(".", "_").replace("/", "_")
        layer_dir = output_dir / layer_name
        layer_dir.mkdir(parents=True, exist_ok=True)

        num_channels = key.shape[1]
        for ch in range(num_channels):
            channel_data = key[0, ch].numpy()
            plt.imshow(
                channel_data,
                cmap="viridis",
                vmin=global_min,
                vmax=global_max,
            )
            plt.axis("off")
            plt.savefig(
                layer_dir / f"channel_{ch}.png",
                format="png",
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close()


def main():
    args = parse_args()

    model = Theia.load_from_checkpoint(args.ckpt)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize(args.res),
            transforms.CenterCrop(args.res),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image = Image.open(args.image).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    keys = extract_keys(model, input_tensor)
    visualize_keys(keys, args.output)


if __name__ == "__main__":
    main()
