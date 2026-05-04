from dataclasses import dataclass

from pathlib import Path
from tqdm import tqdm

from models.linear import SchmidhuberLinear, SchmidhuberLinearConfig
import torch
import wandb


@dataclass
class TrainerConfig:
    model_config: SchmidhuberLinearConfig
    batch_size: int
    num_epochs: int
    learning_rate_predictors: float
    learning_rate_autoencoder: float
    reconstruction_loss_weight: float
    dataset_repo_id: str
    wandb_project: str = "zzsn-projekt"
    wandb_run_name: str | None = None
    wandb_mode: str = "online"  # "online", "offline", or "disabled"
    checkpoint_dir: str


class Trainer:
    def __init__(self, config: TrainerConfig, model: SchmidhuberLinear):
        self.cfg = config
        self.model = model

    def train(self, data_loader: torch.utils.data.DataLoader):
        wandb.init(
            project=self.cfg.wandb_project,
            name=self.cfg.wandb_run_name,
            mode=self.cfg.wandb_mode,
            config={
                "batch_size": self.cfg.batch_size,
                "num_epochs": self.cfg.num_epochs,
                "learning_rate_predictors": self.cfg.learning_rate_predictors,
                "learning_rate_autoencoder": self.cfg.learning_rate_autoencoder,
                "reconstruction_loss_weight": self.cfg.reconstruction_loss_weight,
                "dataset_repo_id": self.cfg.dataset_repo_id,
                "model_config": {
                    "input_dim": self.cfg.model_config.input_dim,
                    "expansion_factor": self.cfg.model_config.expansion_factor,
                    "predictor_hidden_dims": self.cfg.model_config.predictor_hidden_dims,
                    "predictor_dropout": self.cfg.model_config.predictor_dropout,
                    "predictor_embedding_dim": self.cfg.model_config.predictor_embedding_dim,
                },
            },
        )

        mse = torch.nn.MSELoss()
        optimizer_predictors = torch.optim.Adam(
            params=self.model.shared_predictor.parameters(),
            lr=self.cfg.learning_rate_predictors,
        )
        optimizer_autoencoder = torch.optim.Adam(
            params=list(self.model.encoder.parameters())
            + list(self.model.decoder.parameters()),
            lr=self.cfg.learning_rate_autoencoder,
        )
        predictor_losses = []

        # track loss components separately
        reconstruction_losses = []
        predictability_losses = []
        autoencoder_losses = []

        global_step = 0
        try:
            for epoch_idx in tqdm(range(self.cfg.num_epochs), desc="Epochs"):
                # predictors update
                self.model.freeze_autoencoder()
                self.model.unfreeze_predictors()
                for batch in data_loader:
                    xs = batch["activations"]  # (batch_size, input_dim)

                    with torch.no_grad():
                        sparse_representations = self.model.encoder(xs)

                    predictions = self.model.predict_all(sparse_representations)
                    predictor_loss = mse(predictions, sparse_representations)
                    predictor_losses.append(predictor_loss.item())

                    optimizer_predictors.zero_grad()
                    predictor_loss.backward()
                    optimizer_predictors.step()

                    wandb.log(
                        {
                            "train/predictor_loss": predictor_loss.item(),
                            "epoch": epoch_idx,
                        },
                        step=global_step,
                    )
                    global_step += 1

                # autoencoder update
                self.model.freeze_predictors()
                self.model.unfreeze_autoencoder()
                for batch in data_loader:
                    xs = batch["activations"]  # (batch_size, input_dim)

                    sparse_representations = self.model.encoder(xs)
                    reconstructions = self.model.decoder(sparse_representations)
                    predictions = self.model.predict_all(sparse_representations)

                    reconstruction_loss = mse(reconstructions, xs)
                    predictability_loss = mse(predictions, sparse_representations)
                    autoencoder_loss = (
                        self.cfg.reconstruction_loss_weight * reconstruction_loss
                        - predictability_loss
                    )

                    reconstruction_losses.append(reconstruction_loss.item())
                    predictability_losses.append(predictability_loss.item())
                    autoencoder_losses.append(autoencoder_loss.item())

                    optimizer_autoencoder.zero_grad()
                    autoencoder_loss.backward()
                    optimizer_autoencoder.step()

                    wandb.log(
                        {
                            "train/reconstruction_loss": reconstruction_loss.item(),
                            "train/predictability_loss": predictability_loss.item(),
                            "train/autoencoder_loss": autoencoder_loss.item(),
                            "epoch": epoch_idx,
                        },
                        step=global_step,
                    )
                    global_step += 1

                self.save_checkpoint(epoch_idx)

        finally:
            wandb.finish()

        return {
            "predictor_loss": predictor_losses,
            "reconstruction_loss": reconstruction_losses,
            "predictability_loss": predictability_losses,
            "autoencoder_loss": autoencoder_losses,
        }

    def save_checkpoint(self, epoch_idx: int):
        checkpoint_dir = Path(f"{self.cfg.checkpoint_dir}/{wandb.run.id}")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"model_epoch_{epoch_idx}.pt"
        torch.save(self.model.state_dict(), checkpoint_path)

        if self.cfg.wandb_mode != "disabled":
            artifact = wandb.Artifact(
                name=f"model-{wandb.run.id}-epoch_{epoch_idx}", type="model"
            )
            artifact.add_file(str(checkpoint_path))
            wandb.log_artifact(artifact)
