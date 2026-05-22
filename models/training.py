"""Trainer for SchmidhuberLinear model via predictability minimization."""

from dataclasses import asdict, dataclass

from enum import Enum
from pathlib import Path

from torch.utils.data import DataLoader
from tqdm import tqdm

from models.linear import SchmidhuberLinear, SchmidhuberLinearConfig
import torch
import wandb


@dataclass
class TrainerConfig:
    model_config: SchmidhuberLinearConfig
    batches_per_phase: int
    """Number of batches for each training phase (autoencoder or predictor) before switching to the other phase."""
    num_epochs: int
    learning_rate_predictors: float
    learning_rate_autoencoder: float
    reconstruction_loss_weight: float
    checkpoint_dir: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    wandb_project: str = "zzsn-projekt"
    wandb_run_name: str | None = None
    wandb_mode: str = "online"  # "online", "offline", or "disabled"


class TrainingPhase(Enum):
    AUTOENCODER = "autoencoder"
    PREDICTOR = "predictor"


class Trainer:
    def __init__(self, config: TrainerConfig, model: SchmidhuberLinear):
        self.cfg = config
        self.model = model

        self.device = torch.device(self.cfg.device)
        self.model.to(self.device)

        self.loss_fn = torch.nn.MSELoss()
        self.optimizer_predictors = torch.optim.Adam(
            params=self.model.shared_predictor.parameters(),
            lr=self.cfg.learning_rate_predictors,
        )
        self.optimizer_autoencoder = torch.optim.Adam(
            params=list(self.model.encoder.parameters())
            + list(self.model.decoder.parameters()),
            lr=self.cfg.learning_rate_autoencoder,
        )

        self.reconstruction_losses = []
        self.predictability_losses = []
        self.autoencoder_losses = []

        self.val_reconstruction_losses = []
        self.val_predictability_losses = []

        self.global_step = 0
        self.phase = TrainingPhase.AUTOENCODER

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        self.reset()

        wandb.init(
            project=self.cfg.wandb_project,
            name=self.cfg.wandb_run_name,
            mode=self.cfg.wandb_mode,
            config=asdict(self.cfg),
        )

        try:
            for epoch_idx in tqdm(range(self.cfg.num_epochs), desc="Epochs"):
                for batch_idx, batch in enumerate(tqdm(train_loader, desc="Batches")):
                    if batch_idx % self.cfg.batches_per_phase == 0:
                        self.switch_phase()

                    xs = batch["activations"].to(self.device)  # (batch_size, input_dim)

                    if self.phase == TrainingPhase.PREDICTOR:
                        self.predictor_step(xs, epoch_idx)
                    else:
                        self.autoencoder_step(xs, epoch_idx)

                    self.global_step += 1

                self.save_checkpoint(epoch_idx)

        finally:
            wandb.finish()

        return {
            "predictability_loss": self.predictability_losses,
            "reconstruction_loss": self.reconstruction_losses,
            "autoencoder_loss": self.autoencoder_losses,
        }

    def switch_phase(self):
        if self.phase == TrainingPhase.AUTOENCODER:
            self.phase = TrainingPhase.PREDICTOR
            self.model.freeze_autoencoder()
            self.model.unfreeze_predictors()
        else:
            self.phase = TrainingPhase.AUTOENCODER
            self.model.freeze_predictors()
            self.model.unfreeze_autoencoder()

    def predictor_step(self, xs: torch.Tensor, epoch_idx: int):
        with torch.no_grad():
            sparse_representations = self.model.encoder(xs)

        predictions = self.model.predict_all(sparse_representations)
        predictability_loss = self.loss_fn(predictions, sparse_representations)
        self.predictability_losses.append(predictability_loss.item())

        self.optimizer_predictors.zero_grad()
        predictability_loss.backward()
        self.optimizer_predictors.step()

        self.log_losses(epoch_idx, predictability_loss=predictability_loss.item())

    def autoencoder_step(self, xs: torch.Tensor, epoch_idx: int):
        sparse_representations = self.model.encoder(xs)
        reconstructions = self.model.decoder(sparse_representations)
        predictions = self.model.predict_all(sparse_representations)

        reconstruction_loss = self.loss_fn(reconstructions, xs)
        predictability_loss = self.loss_fn(predictions, sparse_representations)
        autoencoder_loss = (
            self.cfg.reconstruction_loss_weight * reconstruction_loss
            - predictability_loss
        )

        self.reconstruction_losses.append(reconstruction_loss.item())
        self.predictability_losses.append(predictability_loss.item())
        self.autoencoder_losses.append(autoencoder_loss.item())

        self.optimizer_autoencoder.zero_grad()
        autoencoder_loss.backward()
        self.optimizer_autoencoder.step()

        self.log_losses(
            epoch_idx,
            reconstruction_loss=reconstruction_loss.item(),
            predictability_loss=predictability_loss.item(),
            autoencoder_loss=autoencoder_loss.item(),
        )

    def reset(self):
        self.reconstruction_losses.clear()
        self.predictability_losses.clear()
        self.autoencoder_losses.clear()
        self.global_step = 0
        self.phase = TrainingPhase.AUTOENCODER

    def log_losses(
        self,
        epoch_idx: int,
        reconstruction_loss: float | None = None,
        predictability_loss: float | None = None,
        autoencoder_loss: float | None = None,
    ):
        wandb.log(
            {
                "train/reconstruction_loss": reconstruction_loss,
                "train/predictability_loss": predictability_loss,
                "train/autoencoder_loss": autoencoder_loss,
                "epoch": epoch_idx,
            },
            step=self.global_step,
        )

    # TODO save model configuration as well
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
