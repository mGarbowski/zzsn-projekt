import pytest
import torch

from datasets import Dataset
from models.linear import SchmidhuberLinear, SchmidhuberLinearConfig
from models.training import Trainer, TrainerConfig


@pytest.fixture
def model_config():
    return SchmidhuberLinearConfig(
        input_dim=20,
        expansion_factor=5,
        predictor_hidden_dims=[16, 16],
        predictor_dropout=0.1,
        predictor_embedding_dim=8,
    )


@pytest.fixture
def model(model_config):
    return SchmidhuberLinear(model_config)


@pytest.fixture
def trainer_config(model_config, tmp_path):
    return TrainerConfig(
        model_config=model_config,
        batches_per_phase=5,
        num_epochs=2,
        learning_rate_predictors=1e-3,
        learning_rate_autoencoder=1e-3,
        reconstruction_loss_weight=1.0,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        wandb_project="TEST",
        wandb_run_name="TEST",
        wandb_mode="disabled",
    )


@pytest.fixture
def data_loader(trainer_config):
    """Random data"""
    activations = torch.randn(10, trainer_config.model_config.input_dim)  # 10 samples
    dataset = Dataset.from_dict({"activations": activations}).with_format("torch")
    return torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)


def test_trainer(trainer_config, data_loader, model):
    """Verify that training loop finishes without errors"""
    trainer = Trainer(trainer_config, model)

    results = trainer.train(data_loader)
    assert results is not None
