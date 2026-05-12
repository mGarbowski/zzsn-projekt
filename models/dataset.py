from dataclasses import dataclass
from datasets import Dataset, load_dataset
import torch


@dataclass(frozen=True)
class DataSourceConfig:
    dataset_repo_id: str = "mgarbowski/zzsn-activations-1"
    batch_size: int = 32
    num_workers: int = 0
    shuffle: bool = True


# TODO change collect activations script so this is not needed
def flatten_token_dataset(dataset_repo_id: str, split: str = "train") -> Dataset:
    """Load dataset and flatten spatial tokens for training.

    Converts shape (256, 1280) per image to 256 individual token samples.
    """
    data = load_dataset(dataset_repo_id, split=split).with_format("pytorch")

    flat_activations = []
    flat_timesteps = []
    flat_prompts = []

    for row in data:
        acts = row["activations"]  # shape: (256, 1280)
        timestep = row["timestep"]
        prompt = row["prompt"]

        for vec in acts:
            flat_activations.append(vec)
            flat_timesteps.append(timestep)
            flat_prompts.append(prompt)

    flat_train = Dataset.from_dict(
        {
            "activations": flat_activations,
            "timestep": flat_timesteps,
            "prompt": flat_prompts,
        }
    ).with_format("torch")
    return flat_train


def get_data_loader(cfg: DataSourceConfig) -> torch.utils.data.DataLoader:
    dataset = flatten_token_dataset(cfg.dataset_repo_id)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers,
    )
