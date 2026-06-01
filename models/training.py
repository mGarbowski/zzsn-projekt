"""Trainer for SchmidhuberLinear model via predictability minimization."""

from dataclasses import asdict, dataclass

from enum import Enum
from pathlib import Path

from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusers import StableDiffusionPipeline
from loguru import logger
from wandb import Artifact

from models.diffusion import GenerationParams, WrappedDiffusion
from models.linear import SchmidhuberLinear, SchmidhuberLinearConfig
import torch
import wandb

_PREVIEW_PROMPT = "monkeys playing poker in photorealistic style"
_PREVIEW_SEED = 0  # num_seeds=1 → seed 0
_PREVIEW_STEPS = 20
_PREVIEW_GUIDANCE = 7.5


@dataclass
class TrainerConfig:
    model_config: SchmidhuberLinearConfig
    batches_per_phase: int
    """Number of batches for each training phase (autoencoder or predictor) before switching to the other phase."""
    num_epochs: int
    num_steps_per_checkpoint: int
    """Number of training steps (mini-batches) between each checkpoint and validation.
    Epochs are not granular enough for this purpose."""
    num_validation_batches_per_checkpoint: int
    """A subset of the validation partition to evaluate the model on after each checkpoint."""
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

        self.global_step = 1
        self.phase = TrainingPhase.AUTOENCODER
        self.wrapped_diffusion: WrappedDiffusion | None = None

    def load_diffusion_model(self, pipeline: StableDiffusionPipeline) -> None:
        """Attach a Stable Diffusion pipeline for checkpoint preview image generation.

        Constructs a WrappedDiffusion around the pipeline and the trainer's own model,
        so there is a single source of truth for the Schmidhuber weights.
        Call this before training if you want preview images logged at each checkpoint.
        """
        self.wrapped_diffusion = WrappedDiffusion(
            pipeline, self.model, layer_name=self.model.cfg.layer_name
        )

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

                    if self.global_step % self.cfg.num_steps_per_checkpoint == 0:
                        self.save_checkpoint(self.global_step)
                        self.validate(val_loader, epoch_idx)
                        self.model.train()

                    self.global_step += 1

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

    def validate(self, val_loader: DataLoader, epoch_idx: int):
        val_reconstruction_losses = []
        val_predictability_losses = []
        val_autoencoder_losses = []

        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(
                tqdm(val_loader, desc="Validation Batches")
            ):
                if batch_idx >= self.cfg.num_validation_batches_per_checkpoint:
                    break

                xs = batch["activations"].to(self.device)

                sparse_representations = self.model.encoder(xs)
                reconstructions = self.model.decoder(sparse_representations)
                predictions = self.model.predict_all(sparse_representations)

                reconstruction_loss = self.loss_fn(reconstructions, xs)
                predictability_loss = self.loss_fn(predictions, sparse_representations)
                autoencoder_loss = (
                    self.cfg.reconstruction_loss_weight * reconstruction_loss
                    - predictability_loss
                )

                val_reconstruction_losses.append(reconstruction_loss.item())
                val_predictability_losses.append(predictability_loss.item())
                val_autoencoder_losses.append(autoencoder_loss.item())

        mean_reconstruction_loss = sum(val_reconstruction_losses) / len(
            val_reconstruction_losses
        )
        mean_predictability_loss = sum(val_predictability_losses) / len(
            val_predictability_losses
        )
        mean_autoencoder_loss = sum(val_autoencoder_losses) / len(
            val_autoencoder_losses
        )

        self.log_losses(
            epoch_idx,
            reconstruction_loss=mean_reconstruction_loss,
            predictability_loss=mean_predictability_loss,
            autoencoder_loss=mean_autoencoder_loss,
            mode="val",
        )

    def reset(self):
        self.reconstruction_losses.clear()
        self.predictability_losses.clear()
        self.autoencoder_losses.clear()
        self.global_step = 1
        self.phase = TrainingPhase.AUTOENCODER

    def log_losses(
        self,
        epoch_idx: int,
        reconstruction_loss: float | None = None,
        predictability_loss: float | None = None,
        autoencoder_loss: float | None = None,
        mode: str = "train",
    ):
        wandb.log(
            {
                f"{mode}/reconstruction_loss": reconstruction_loss,
                f"{mode}/predictability_loss": predictability_loss,
                f"{mode}/autoencoder_loss": autoencoder_loss,
                "epoch": epoch_idx,
            },
            step=self.global_step,
        )

    def save_checkpoint(self, step_idx: int):
        checkpoint_dir = Path(f"{self.cfg.checkpoint_dir}/{wandb.run.id}")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"model_step_{step_idx}.pt"
        torch.save(self.model.state_dict(), checkpoint_path)

        if self.cfg.wandb_mode != "disabled":
            artifact = wandb.Artifact(
                name=f"model-{wandb.run.id}-step_{step_idx}", type="model"
            )
            artifact.add_file(str(checkpoint_path))

            if self.wrapped_diffusion is not None:
                self.generate_preview_images(artifact, checkpoint_dir, step_idx)

            wandb.log_artifact(artifact)

    def generate_preview_images(
        self, artifact: Artifact, checkpoint_dir: Path, step_idx: int
    ) -> None:
        """Generate a sample image with and without intervention and add to the artifact"""
        assert self.wrapped_diffusion is not None

        preview_params = GenerationParams(
            prompts=[_PREVIEW_PROMPT],
            num_seeds=1,
            num_inference_steps=_PREVIEW_STEPS,
            guidance_scale=_PREVIEW_GUIDANCE,
        )

        normal_img = self.wrapped_diffusion.generate(preview_params)[0].image
        intervened_img = self.wrapped_diffusion.generate_with_intervention(
            preview_params, {}
        )[0].image

        normal_path = checkpoint_dir / f"preview_normal_step_{step_idx}.png"
        intervened_path = checkpoint_dir / f"preview_intervention_step_{step_idx}.png"
        normal_img.save(str(normal_path))
        intervened_img.save(str(intervened_path))

        artifact.add_file(str(normal_path))
        artifact.add_file(str(intervened_path))
