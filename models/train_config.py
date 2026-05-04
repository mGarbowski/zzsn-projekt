from dataclasses import dataclass, field
from typing import Optional

from hydra.core.config_store import ConfigStore

from conf.paths import SCRATCH_ROOT


@dataclass
class TrainScriptConfig:
    # model config
    input_dim: int = 1280
    expansion_factor: int = 2
    predictor_hidden_dims: list[int] = field(default_factory=lambda: [256])
    predictor_dropout: float = 0.3
    predictor_embedding_dim: int = 128

    # trainer config
    batch_size: int = 32
    num_epochs: int = 3
    learning_rate_predictors: float = 4e-4
    learning_rate_autoencoder: float = 1e-4
    reconstruction_loss_weight: float = 1.0

    # data config
    dataset_repo_id: str = "mgarbowski/zzsn-activations-1"
    dataset_split: str = "train"
    num_workers: int = 0
    shuffle: bool = True

    # runtime
    seed: int = 42
    device: str = "cuda"

    # logging
    wandb_project: str = "zzsn-projekt"
    wandb_run_name: Optional[str] = None
    wandb_mode: str = "online"  # online | offline | disabled
    checkpoint_dir: Optional[str] = None

    def __post_init__(self) -> None:
        if self.checkpoint_dir is None:
            self.checkpoint_dir = str(SCRATCH_ROOT / "checkpoints")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be > 0")
        if self.learning_rate_predictors <= 0:
            raise ValueError("learning_rate_predictors must be > 0")
        if self.learning_rate_autoencoder <= 0:
            raise ValueError("learning_rate_autoencoder must be > 0")
        if self.reconstruction_loss_weight < 0:
            raise ValueError("relative reconstruction_loss_weight must be >= 0")
        if self.wandb_mode not in {"online", "offline", "disabled"}:
            raise ValueError("wandb_mode must be one of: online, offline, disabled")


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="train_config", node=TrainScriptConfig)
