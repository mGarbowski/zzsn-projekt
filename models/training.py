from dataclasses import dataclass

from tqdm import tqdm

from models.linear import SchmidhuberLinear, SchmidhuberLinearConfig
import torch

@dataclass
class TrainerConfig:
    model_config: SchmidhuberLinearConfig
    batch_size: int
    num_epochs: int
    learning_rate_predictors: float
    learning_rate_autoencoder: float
    reconstruction_loss_weight: float
    dataset_repo_id: str


class Trainer:
    def __init__(self, config: TrainerConfig, model: SchmidhuberLinear):
        self.cfg = config
        self.model = model
        

    def train(self, data_loader: torch.utils.data.DataLoader):
        
        mse = torch.nn.MSELoss()
        optimizer_predictors = torch.optim.Adam(
            params=self.model.shared_predictor.parameters(),
            lr=self.cfg.learning_rate_predictors,
        )
        optimizer_autoencoder = torch.optim.Adam(
            params=list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()),
            lr=self.cfg.learning_rate_autoencoder,
        )
        predictor_losses = []

        # track loss components separately
        reconstruction_losses = []
        predictability_losses = []
        autoencoder_losses = []
        
        for epoch_idx in tqdm(range(self.cfg.num_epochs), desc="Epochs"):
        
            # predictors update
            self.model.freeze_autoencoder()
            self.model.unfreeze_predictors()
            for i, batch in enumerate(data_loader):
                print(f"{i}/{len(data_loader)}")
                xs = batch["activations"]  # (batch_size, input_dim)
                
                with torch.no_grad():
                    sparse_representations = self.model.encoder(xs)

                predictions = self.model.predict_all(sparse_representations)
                predictor_loss = mse(predictions, sparse_representations)
                predictor_losses.append(predictor_loss.item())

                predictor_loss.backward()
                optimizer_predictors.step()
                optimizer_predictors.zero_grad()

            # autoencoder update
            self.model.freeze_predictors()
            self.model.unfreeze_autoencoder()
            for i, batch in enumerate(data_loader):
                print(f"{i}/{len(data_loader)}")
                xs = batch["activations"]  # (batch_size, input_dim)

                sparse_representations = self.model.encoder(xs)
                reconstructions = self.model.decoder(sparse_representations)
                predictions = self.model.predict_all(sparse_representations)

                reconstruction_loss = mse(reconstructions, xs)
                predictability_loss = mse(predictions, xs)
                autoencoder_loss = self.cfg.reconstruction_loss_weight * reconstruction_loss - predictability_loss

                reconstruction_losses.append(reconstruction_loss.item())
                predictability_losses.append(predictability_loss.item())
                autoencoder_losses.append(autoencoder_loss.item())

                autoencoder_loss.backward()
                optimizer_autoencoder.step()
                optimizer_autoencoder.zero_grad()
                
        return {
            "predictor_loss": predictor_losses,
            "reconstruction_loss": reconstruction_losses,
            "predictability_loss": predictability_losses,
            "autoencoder_loss": autoencoder_losses,
        }
