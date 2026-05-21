"""Loading the dataset with Stable Diffusion activations from HuggingFace"""

from dataclasses import dataclass
from datasets import load_dataset
import torch


@dataclass(frozen=True)
class DataSourceConfig:
    dataset_repo_id: str = "mgarbowski/zzsn-activations-1_unet.up_blocks.1.attentions.2"
    dataset_split: str = "train"
    batch_size: int = 256
    num_workers: int = 0
    shuffle: bool = True


def get_data_loader(cfg: DataSourceConfig) -> torch.utils.data.DataLoader:
    dataset = load_dataset(cfg.dataset_repo_id, split=cfg.dataset_split).with_format(
        "pytorch"
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers,
    )
