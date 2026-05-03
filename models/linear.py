
from dataclasses import dataclass

from torch import nn
import torch


@dataclass
class SchmidhuberLinearConfig:
    input_dim: int
    expansion_factor: int
    predictor_hidden_dims: list[int]
    predictor_dropout: float

class SchmidhuberLinear(nn.Module):
    """A simple linear variant of Schmidhuber's architecture for predictability minimization.
    
    Sa imple case where encoder is linear.
    Uses decoder for reconstructing the input from the sparse representation (to allow interventions).
    """

    def __init__(self, config: SchmidhuberLinearConfig):
        super().__init__()
        self.cfg = config
        self.dictionary_dim = config.input_dim * config.expansion_factor  # dimension of the sparse representation
        self.encoder = nn.Linear(in_features=config.input_dim, out_features=self.dictionary_dim)
        self.decoder = nn.Linear(in_features=self.dictionary_dim, out_features=config.input_dim)

        # One predictor per dimension of the sparse representation, each trying to predict that dimension from the others
        self.predictors = nn.ModuleList([
            SchmidhuberPredictor(
                dictionary_dim=self.dictionary_dim,
                mlp_hidden_dims=config.predictor_hidden_dims,
                dropout=config.predictor_dropout,
            )
            for _ in range(self.dictionary_dim)
        ])

    def predict_kth(self, k: int, sparse_representation: torch.Tensor) -> torch.Tensor:
        """Predict the k-th dimension of the sparse representation from the others."""
        assert len(sparse_representation.shape) == 2, "Expected input shape (batch_dim, dictionary_dim)"
        assert sparse_representation.shape[1] == self.dictionary_dim, "Expected input shape (batch_dim, dictionary_dim)"
        assert 0 <= k < self.dictionary_dim
        # Batch without the k-th dimension for each sparse vector in batch
        input_to_predictor = torch.cat([sparse_representation[:, :k], sparse_representation[:, k+1:]], dim=1)
        return self.predictors[k](input_to_predictor)
    
    def predict_all(self, sparse_representation: torch.Tensor) -> torch.Tensor:
        """Predict all dimensions using corresponding predictors.
        
        Takes in a batch of sparse representations (batch_dim, dictionary_dim).
        Return batch of predictions (batch_dim, dictionary_dim)
        """
        return torch.cat(
            [
                self.predict_kth(k, sparse_representation)
                for k in range(self.dictionary_dim)
            ],
            dim=1
        )


class SchmidhuberPredictor(nn.Module):
    def __init__(self, dictionary_dim: int, mlp_hidden_dims: list[int], dropout: float):
        super().__init__()
        self.net = MLP(
            input_dim=dictionary_dim-1,
            hidden_dims=mlp_hidden_dims,
            output_dim=1,
            dropout=dropout,
        )

    def forward(self, x):
        return self.net(x)


# TODO use existing implementation
class MLP(nn.Module):
    """Multilayer perceptron with ReLU, dropout and batch normalization."""
    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int, dropout: float):
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