"""Loading the dataset with Stable Diffusion activations from HuggingFace"""

from dataclasses import dataclass
from typing import TypedDict

from datasets import load_dataset, DatasetDict
import torch
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class DataSourceConfig:
    dataset_repo_id: str = "mgarbowski/zzsn-activations-1_unet.up_blocks.1.attentions.2"
    dataset_split: str = "train"
    batch_size: int = 256
    num_workers: int = 0
    shuffle: bool = True


class DataLoadersDict(TypedDict):
    train: DataLoader
    val: DataLoader

def get_data_loaders(cfg: DataSourceConfig) -> DataLoadersDict:
    """Split the dataset and create data loaders

    only a single split is saved on HF
    """
    dataset = load_dataset(cfg.dataset_repo_id, split=cfg.dataset_split).with_format(
        "pytorch"
    )
    split = dataset.train_test_split(test_size=0.2, seed=42)
    new_dataset = DatasetDict({"train": split["train"], "val": split["test"]})

    train_loader = DataLoader(
        new_dataset["train"],
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        new_dataset["val"],
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    return {"train": train_loader, "val": val_loader}
