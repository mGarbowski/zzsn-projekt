# scripts/train.py
import sys
import traceback
from pathlib import Path

import hydra
import torch

from omegaconf import OmegaConf, SCMode

from conf.paths import HYDRA_CONFIG_ROOT_STR
from models.dataset import DataSourceConfig, get_data_loader
from models.linear import SchmidhuberLinear, SchmidhuberLinearConfig
from models.train_config import TrainScriptConfig, register_configs
from models.training import Trainer, TrainerConfig

register_configs()


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

        data_source_cfg = DataSourceConfig(
            dataset_repo_id=run_cfg.dataset_repo_id,
            batch_size=run_cfg.batch_size,
            num_workers=run_cfg.num_workers,
            shuffle=run_cfg.shuffle,
        )

        model_cfg = SchmidhuberLinearConfig(
            input_dim=run_cfg.input_dim,
            expansion_factor=run_cfg.expansion_factor,
            predictor_hidden_dims=run_cfg.predictor_hidden_dims,
            predictor_dropout=run_cfg.predictor_dropout,
            predictor_embedding_dim=run_cfg.predictor_embedding_dim,
        )

        trainer_cfg = TrainerConfig(
            model_config=model_cfg,
            batches_per_phase=run_cfg.batches_per_phase,
            num_epochs=run_cfg.num_epochs,
            learning_rate_predictors=run_cfg.learning_rate_predictors,
            learning_rate_autoencoder=run_cfg.learning_rate_autoencoder,
            reconstruction_loss_weight=run_cfg.reconstruction_loss_weight,
            wandb_project=run_cfg.wandb_project,
            wandb_run_name=run_cfg.wandb_run_name,
            wandb_mode=run_cfg.wandb_mode,
            checkpoint_dir=Path(run_cfg.checkpoint_dir),
            device=run_cfg.device,
        )

        model = SchmidhuberLinear(model_cfg)
        model = model.to(torch.device(run_cfg.device))
        print(f"Created model with {model.num_parameters()} parameters")

        print("Loading data...")
        loader = get_data_loader(data_source_cfg)

        print("Starting training...")
        Trainer(trainer_cfg, model).train(loader)

    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
