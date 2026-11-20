import argparse
from dataclasses import asdict

import lightning.pytorch as pl
from lightning.pytorch import loggers
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from configs import CONFIGS
from data_module import DataModule
from model import Theia
from utils.ema import EMA, EMAModelCheckpoint


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        choices=CONFIGS.keys(),
        required=True,
        help="Specify which model configuration to use.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Root path to the dataset. Must contain folder structures expected by torchvision.datasets.ImageFolder.",
    )
    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=16,
        help="Number of processes used for the PyTorch dataloader.",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default="models/",
        help="Directory to save the model checkpoints from each epoch.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/",
        help="Directory to save the training logs.",
    )
    parser.add_argument(
        "--debug",
        type=int,
        default=0,
        help="Run n batches of training and validation for debug. Set to 0 to disable.",
    )
    parser.add_argument(
        "--ft-ckpt",
        type=str,
        default=None,
        help="Path to the checkpoint file for fine-tuning. Must be provided if using fine-tuning.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to the checkpoint file to resume training from. Will load EMA weights if available.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config = CONFIGS[args.config]()

    data = DataModule(
        dataset_dir=args.dataset_dir,
        dataloader_num_workers=args.dataloader_num_workers,
        **asdict(config),
    )

    if "ft" in args.config:
        if args.ft_ckpt is None:
            raise ValueError("Fine-tuning checkpoint path must be provided.")
        model = Theia.load_from_checkpoint(args.ft_ckpt, **asdict(config))
    else:
        model = Theia(**asdict(config))

    callbacks = [
        LearningRateMonitor(logging_interval="epoch"),
    ]
    if config.ema > 0:
        callbacks.extend(
            [
                EMA(
                    decay=config.ema,
                    apply_ema_every_n_steps=config.ema_steps,
                    start_step=config.ema_start_step,
                ),
                EMAModelCheckpoint(dirpath=args.ckpt_dir, save_top_k=-1),
            ]
        )
    else:
        callbacks.append(ModelCheckpoint(dirpath=args.ckpt_dir, save_top_k=-1))

    logger = loggers.CSVLogger(
        args.log_dir,
        name=args.config,
    )

    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        precision="32-true",
        fast_dev_run=False if args.debug == 0 else args.debug,
        max_epochs=config.max_epochs,
        num_sanity_val_steps=0,
        gradient_clip_val=1.0,
        sync_batchnorm=True,
    )

    trainer.fit(model, datamodule=data, ckpt_path=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
