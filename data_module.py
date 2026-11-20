import lightning.pytorch as pl
import torch
import torchvision.transforms.v2 as transforms
from timm.data import Mixup
from torch.utils.data import DataLoader, default_collate
from torchvision.datasets import ImageFolder


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_dir: str,
        num_classes: int,
        train_batch_size: int,
        train_image_size: int,
        color_jitter: float,
        mixup_alpha: float,
        cutmix_alpha: float,
        label_smoothing: float,
        eval_batch_size: int,
        eval_image_size: int,
        eval_ratio: float,
        dataloader_num_workers: int,
        loss: str = "bce",
        # skip any model parameters
        **kwargs,
    ) -> None:
        """
        DataModule for handling dataset loading, preprocessing, and augmentation.

        Args:
            dataset_dir (str): Path to the dataset directory.
            num_classes (int): Number of classes in the dataset.
            train_batch_size (int): Batch size for training.
            train_image_size (int): Size to resize training images to.
            color_jitter (float): Color jitter intensity.
            mixup_alpha (float): Alpha parameter for mixup augmentation.
            cutmix_alpha (float): Alpha parameter for cutmix augmentation.
            label_smoothing (float): Label smoothing factor.
            eval_batch_size (int): Batch size for evaluation.
            eval_image_size (int): Size to crop evaluation images to.
            eval_ratio (float): Crop ratio for evaluation images.
            dataloader_num_workers (int): Number of workers for data loader.
            loss (str): Training loss, "ce" or "bce".
        """
        super().__init__()
        self.dataset = dataset_dir
        self.num_classes = num_classes

        self.train_batch_size = train_batch_size
        self.train_image_size = train_image_size
        self.color_jitter = color_jitter
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.label_smoothing = label_smoothing
        self.loss = loss

        self.eval_batch_size = eval_batch_size
        self.eval_image_size = eval_image_size
        self.eval_ratio = eval_ratio

        self.num_workers = dataloader_num_workers

    def train_dataloader(self) -> DataLoader:
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    self.train_image_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.TrivialAugmentWide(
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ColorJitter(
                    self.color_jitter, self.color_jitter, self.color_jitter
                ),
                transforms.PILToTensor(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        dataset = ImageFolder(self.dataset + "/train", train_transform)

        mixup_fn = Mixup(
            mixup_alpha=self.mixup_alpha,
            cutmix_alpha=self.cutmix_alpha,
            label_smoothing=self.label_smoothing,
            num_classes=self.num_classes,
        )

        def collate_fn(batch):
            samples, targets = default_collate(batch)
            samples, targets = mixup_fn(samples, targets)

            if self.loss == "bce":
                targets = targets.gt(0.0).type(targets.dtype)

            return samples, targets

        return DataLoader(
            dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        valid_transform = transforms.Compose(
            [
                transforms.Resize(
                    int(self.eval_image_size // self.eval_ratio),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.CenterCrop(self.eval_image_size),
                transforms.PILToTensor(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        dataset = ImageFolder(
            self.dataset + "/val",
            valid_transform,
        )

        return DataLoader(
            dataset,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
        )
