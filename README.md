# Theia

## About

Theia follows each mobile inverted bottleneck with two convolutional Attention Free Transformer encoders. Each encoder applies global response normalization to queries, keys, values, and the MLP's hidden activations.

## Installation

To set up the environment on GPU/CPU, run:

```bash
pip install -r requirements.txt
```

To set up the environment on a TPU VM, run:

```bash
sudo apt-get install libopenblas-dev -y
pip install -r requirements-tpu.txt
```

## Pretrained Models

The Theia family was trained on the [ImageNet-1k dataset](https://www.image-net.org/challenges/LSVRC/2012/2012-downloads.php). All checkpoint files provided are stateful. For more information, refer to [PyTorch Lightning's documentation](https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html#contents-of-a-checkpoint).

| Name        | Input Res. | Top-1      | Top-5      | Loss Function | EMA | Epoch | Validation Loss | Training Loss | Download                                                                                               | Log                                                                                                              |
| ----------- | ---------- | ---------- | ---------- | ------------- | --- | ----- | --------------- | ------------- | ------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| Theia Tiny  | 224²       | 85.03%     | 96.67%     | CE            | Yes | 345   | 0.8493          | 2.4795        | [Link](https://archive.org/download/theiacheckpoints/Theia.zip/Theia%2FCE%2FTheia%20Tiny.ckpt "Unaveraged checkpoint")         | [Link](https://archive.org/download/theiacheckpoints/Theia.zip/Theia%2FCE%2FTheia%20Tiny%20Training%20Log.csv)   |
| Theia Tiny  | 224²       | 84.01%     | 96.23%     | BCE           | Yes | 398   | 0.8307          | 4.3416        | [Link](https://archive.org/download/theiacheckpoints/Theia.zip/Theia%2FBCE%2FTheia%20Tiny%20EMA.ckpt)  | [Link](https://archive.org/download/theiacheckpoints/Theia.zip/Theia%2FBCE%2FTheia%20Tiny%20Training%20Log.csv)  |
| Theia Tiny  | 224²       | 84.13%     | 95.65%     | BCE           | No  | 376   | 0.8651          | 4.4000        | [Link](https://archive.org/download/theiacheckpoints/Theia.zip/Theia%2FBCE%2FTheia%20Tiny.ckpt)        | [Link](https://archive.org/download/theiacheckpoints/Theia.zip/Theia%2FBCE%2FTheia%20Tiny%20Training%20Log.csv)  |
| Theia Small | 224²       | 85.48%     | 97.31%     | CE            | Yes | 264   | 0.8086          | 2.6904        | [Link](https://archive.org/download/theiacheckpoints/Theia.zip/Theia%2FCE%2FTheia%20Small.ckpt)        | [Link](https://archive.org/download/theiacheckpoints/Theia.zip/Theia%2FCE%2FTheia%20Small%20Training%20Log.csv)  |
| Theia Small | 224²       | 85.67%     | 97.12%     | BCE           | Yes | 351   | 0.7774          | 3.9484        | [Link](https://archive.org/download/theiacheckpoints/Theia.zip/Theia%2FBCE%2FTheia%20Small%20EMA.ckpt) | [Link](https://archive.org/download/theiacheckpoints/Theia.zip/Theia%2FBCE%2FTheia%20Small%20Training%20Log.csv) |
| Theia Small | 224²       | 84.96%     | 96.93%     | BCE           | No  | 346   | 0.9186          | 3.9941        | [Link](https://archive.org/download/theiacheckpoints/Theia.zip/Theia%2FBCE%2FTheia%20Small.ckpt)       | [Link](https://archive.org/download/theiacheckpoints/Theia.zip/Theia%2FBCE%2FTheia%20Small%20Training%20Log.csv) |
| Theia Base  | 224²       | 86.31%     | **97.50%** | CE            | Yes | 279   | 0.7769          | 2.4255        | [Link](https://archive.org/download/theiacheckpoints/Theia.zip/Theia%2FCE%2FTheia%20Base.ckpt "Unaveraged checkpoint")         | [Link](https://archive.org/download/theiacheckpoints/Theia.zip/Theia%2FCE%2FTheia%20Base%20Training%20Log.csv)   |
| Theia Base  | 224²       | 86.12%     | 96.99%     | BCE           | Yes | 337   | 0.7914          | 3.4640        | [Link](https://archive.org/download/theiacheckpoints/Theia.zip/Theia%2FBCE%2FTheia%20Base%20EMA.ckpt)  | [Link](https://archive.org/download/theiacheckpoints/Theia.zip/Theia%2FBCE%2FTheia%20Base%20Training%20Log.csv)  |
| Theia Base  | 224²       | 84.71%     | 96.23%     | BCE           | No  | 310   | 0.8857          | 4.0137        | [Link](https://archive.org/download/theiacheckpoints/Theia.zip/Theia%2FBCE%2FTheia%20Base.ckpt)        | [Link](https://archive.org/download/theiacheckpoints/Theia.zip/Theia%2FBCE%2FTheia%20Base%20Training%20Log.csv)  |
| Theia Large | 224²       | **86.44%** | 96.87%     | BCE           | Yes | 313   | 0.7674          | 3.5877        | [Link](https://archive.org/download/theiacheckpoints/Theia.zip/Theia%2FBCE%2FTheia%20Large%20EMA.ckpt) | [Link](https://archive.org/download/theiacheckpoints/Theia.zip/Theia%2FBCE%2FTheia%20Large%20Training%20Log.csv) |
| Theia Large | 224²       | 85.28%     | 96.03%     | BCE           | No  | 304   | 0.8495          | 3.7742        | [Link](https://archive.org/download/theiacheckpoints/Theia.zip/Theia%2FBCE%2FTheia%20Large.ckpt)       | [Link](https://archive.org/download/theiacheckpoints/Theia.zip/Theia%2FBCE%2FTheia%20Large%20Training%20Log.csv) |

Any intermediate checkpoints can be requested.

## Training and Transfer Learning

To set up the ImageNet-1k dataset, download both the training and validation sets. Use this [script](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4) to extract and organize the dataset.

We provide [train.py](train.py), which contains all the necessary code to train a Theia model and log the training progress. All hyperparameters are set in [model_config.py](model_config.py). The code is also used to fine-tune the model at a higher resolution, the fine-tuning hyperparameters are set in [fine_tuning_config.py](fine_tuning_config.py).

## Inference

Here's a sample code snippet to perform inference with Theia:

```python
import torch

from model import Theia

model = Theia.load_from_checkpoint("path_to.ckpt")
model.eval()

with torch.no_grad():
    sample = torch.rand(3, 224, 224)  # Channel, Width, Height
    sample = sample.unsqueeze(0)      # Batch, Channel, Width, Height
    pred = model(sample)              # Batch, # of class
```

To help analyze and interpret how the model processes images, we provide a visualization tool for the key of each AFTConv2d layer. You can visualize the key for any input image using [visualize_key.py](visualize_key.py).

## Credits

Our research was supported by Cloud TPUs from Google's [TPU Research Cloud (TRC)](https://sites.research.google/trc/about/).
