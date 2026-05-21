"""Implementation of the Schmidhuber model.

Linear encoder and decoder.
Shared predictor, masking the predicted dimension from the input,
embedding of the predicted dimension index concatenated to the input.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import wandb
from torch import nn


@dataclass
class SchmidhuberLinearConfig:
    input_dim: int
    expansion_factor: int
    predictor_hidden_dims: list[int]
    predictor_dropout: float
    predictor_embedding_dim: int
    layer_name: str = "unet.up_blocks.1.attentions.2"


class SchmidhuberLinear(nn.Module):
    """A simple linear variant of Schmidhuber's architecture for predictability minimization.

    Simple case where encoder is linear.
    Uses decoder for reconstructing the input from the sparse representation (to allow interventions).
    """

    @classmethod
    def from_wandb_artifact(cls, artifact_id: str, device: str = "cpu"):
        """Instantiate a SchmidhuberLinear from a W&B model artifact.

        The artifact's run config must contain a 'model_config' key.

        Args:
            artifact_id: W&B artifact identifier, e.g.
                         "entity/project/model-{run_id}-epoch_9:latest"
            device: device to load the model onto.
        """
        api = wandb.Api()
        artifact = api.artifact(artifact_id, type="model")
        run = artifact.logged_by()
        config = SchmidhuberLinearConfig(**run.config["model_config"])

        model = cls(config)
        artifact_dir = Path(artifact.download())
        checkpoint_path = next(artifact_dir.glob("*.pt"))
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
        return model

    def __init__(self, config: SchmidhuberLinearConfig):
        super().__init__()
        self.cfg = config
        self.dictionary_dim = (
            config.input_dim * config.expansion_factor
        )  # dimension of the sparse representation
        self.encoder = nn.Linear(
            in_features=config.input_dim, out_features=self.dictionary_dim
        )
        self.decoder = nn.Linear(
            in_features=self.dictionary_dim, out_features=config.input_dim
        )

        # Same predictor reused for all dimension with masking
        self.shared_predictor = SchmidhuberSharedPredictor(
            dictionary_dim=self.dictionary_dim,
            mlp_hidden_dims=config.predictor_hidden_dims,
            mlp_dropout=config.predictor_dropout,
            index_embedding_dim=config.predictor_embedding_dim,
        )

    def predict_all(self, sparse_representation: torch.Tensor) -> torch.Tensor:
        """Predict all dimensions using the shared predictor.

        Takes in a batch of sparse representations (batch_dim, dictionary_dim).
        Returns a batch of predictions (batch_dim, dictionary_dim).

        The predictor takes in the sparse vector and dimension index k (from range [0, dictionary_dim-1]),
        masks the k-th element of the sparse vector and predicts its value based on the rest of the vector and the dimension index.

        This calculation is batched, a single pass computes the predictions for all dimensions for all sparse vector in the mini-batch.
        """
        batch_size, _ = sparse_representation.shape
        device = sparse_representation.device

        # Repeat each sparse vector dictionary_dim times
        # [x1, x1, ..., x1, x2, x2, ..., x2, ...]
        # Shape (batch_size * dictionary_dim, dictionary_dim)
        repeated = sparse_representation.repeat_interleave(self.dictionary_dim, dim=0)

        # Index of the dimension to mask and predict
        # [0, 1, 2, ..., D-1, 0, 1, 2, ..., D-1, ...]
        # Shape (batch_size * dictionary_dim,)
        predicted_dim_idx = torch.arange(self.dictionary_dim, device=device).repeat(
            batch_size
        )

        preds = self.shared_predictor(repeated, predicted_dim_idx)  # (B*D, 1)
        preds = preds.squeeze(-1).view(batch_size, self.dictionary_dim)  # (B, D)
        return preds

    def num_parameters(self) -> int:
        """Return the total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())

    def freeze_autoencoder(self):
        """Freeze the encoder and decoder weights (for predictors update)."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        self.decoder.eval()

    def unfreeze_autoencoder(self):
        """Unfreeze the encoder and decoder weights (for autoencoder update)."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        for param in self.decoder.parameters():
            param.requires_grad = True
        self.encoder.train()
        self.decoder.train()

    def freeze_predictors(self):
        """Freeze the predictor weights (for autoencoder update)."""
        for param in self.shared_predictor.parameters():
            param.requires_grad = False
        self.shared_predictor.eval()

    def unfreeze_predictors(self):
        """Unfreeze the predictor weights (for predictors update)."""
        for param in self.shared_predictor.parameters():
            param.requires_grad = True
        self.shared_predictor.train()


class SchmidhuberSharedPredictor(nn.Module):
    """One network is reused for predicting each of the dictionary dimensions

    Backbone is an MLP
    the MLP receives as input the sparse representation vector with the predicted dimension zeroed out,
    concatenated with an embedding of the predicted dimension index (to allow the network to know which dimension is being predicted).

    the output is scalar - the prediction for the value of the predicted dimension.
    """

    def __init__(
        self,
        dictionary_dim: int,
        mlp_hidden_dims: list[int],
        mlp_dropout: float,
        index_embedding_dim: int,
    ):
        super().__init__()
        self.dictionary_dim = dictionary_dim
        self.index_embedding = nn.Embedding(
            num_embeddings=dictionary_dim, embedding_dim=index_embedding_dim
        )
        self.mlp = MLP(
            input_dim=dictionary_dim
            + index_embedding_dim,  # sparse representation with one dimension zeroed out + index embedding
            hidden_dims=mlp_hidden_dims,
            output_dim=1,  # scalar prediction for the value of the predicted dimension
            dropout=mlp_dropout,
        )

    def forward(
        self, sparse_representation: torch.Tensor, predicted_dim_idx: torch.Tensor
    ):
        """Predict one element from the sparse_representation based on the rest.
        The predicted dimension is zeroed out before prediction.

        sparse representation: (batch_dim, dictionary_dim)
        predicted_dim_idx: (batch_dim,) - index of dimension to predict for each batch item
        """
        assert sparse_representation.shape[0] == predicted_dim_idx.shape[0]
        assert len(sparse_representation.shape) == 2
        assert len(predicted_dim_idx.shape) == 1

        batch_size = predicted_dim_idx.shape[0]
        masked_representation = sparse_representation.clone()
        idx = torch.arange(batch_size, device=sparse_representation.device)
        masked_representation[idx, predicted_dim_idx] = (
            0.0  # zero out the predicted dimension
        )
        emb = self.index_embedding(predicted_dim_idx)
        mlp_input = torch.cat([masked_representation, emb], dim=-1)
        return self.mlp(mlp_input)


class MLP(nn.Module):
    """Multilayer perceptron with ReLU, dropout and batch normalization."""

    def __init__(
        self, input_dim: int, hidden_dims: list[int], output_dim: int, dropout: float
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout = dropout

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=self.dropout))
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=self.dropout))
        layers.append(nn.Linear(hidden_dims[-1], output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
