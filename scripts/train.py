# scripts/train.py
import sys
import traceback
from pathlib import Path

import hydra
import torch
from datasets import Dataset, load_dataset
from omegaconf import OmegaConf, SCMode

from conf.paths import HYDRA_CONFIG_ROOT_STR
from models.linear import SchmidhuberLinear, SchmidhuberLinearConfig
from models.train_config import TrainScriptConfig, register_configs
from models.training import Trainer, TrainerConfig

register_configs()


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


@hydra.main(
    version_base=None,
    config_name="train",
    config_path=HYDRA_CONFIG_ROOT_STR,
)
def main(cfg: TrainScriptConfig) -> None:
    run_cfg: TrainScriptConfig = OmegaConf.to_container(
        cfg, resolve=True, structured_config_mode=SCMode.INSTANTIATE
    )
    try:
        torch.manual_seed(run_cfg.seed)

        model_cfg = SchmidhuberLinearConfig(
            input_dim=run_cfg.input_dim,
            expansion_factor=run_cfg.expansion_factor,
            predictor_hidden_dims=run_cfg.predictor_hidden_dims,
            predictor_dropout=run_cfg.predictor_dropout,
            predictor_embedding_dim=run_cfg.predictor_embedding_dim,
        )

        trainer_cfg = TrainerConfig(
            model_config=model_cfg,
            batch_size=run_cfg.batch_size,
            num_epochs=run_cfg.num_epochs,
            learning_rate_predictors=run_cfg.learning_rate_predictors,
            learning_rate_autoencoder=run_cfg.learning_rate_autoencoder,
            reconstruction_loss_weight=run_cfg.reconstruction_loss_weight,
            dataset_repo_id=run_cfg.dataset_repo_id,
            wandb_project=run_cfg.wandb_project,
            wandb_run_name=run_cfg.wandb_run_name,
            wandb_mode=run_cfg.wandb_mode,
            checkpoint_dir=Path(run_cfg.checkpoint_dir),
            device=run_cfg.device,
        )

        model = SchmidhuberLinear(model_cfg)
        model = model.to(torch.device(run_cfg.device))
        print(f"Created model with {model.num_parameters()} parameters")

        flat_train = flatten_token_dataset(
            run_cfg.dataset_repo_id, run_cfg.dataset_split
        )

        loader = torch.utils.data.DataLoader(
            flat_train,
            batch_size=run_cfg.batch_size,
            shuffle=run_cfg.shuffle,
            num_workers=run_cfg.num_workers,
        )

        print("Starting training...")
        Trainer(trainer_cfg, model).train(loader)

    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
