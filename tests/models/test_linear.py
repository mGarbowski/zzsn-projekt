import pytest
import torch

from models.linear import SchmidhuberLinear, SchmidhuberLinearConfig


@pytest.fixture
def config():
    return SchmidhuberLinearConfig(
        input_dim=16,
        expansion_factor=4,
        predictor_dropout=0.1,
        predictor_hidden_dims=[16, 16]
    )

@pytest.fixture
def model(config):
    return SchmidhuberLinear(config)


@pytest.fixture
def batch_size():
    return 10

@pytest.fixture
def batch(config, batch_size):
    return torch.rand((batch_size, config.input_dim))

class TestSchmidhuberLinear:
    def test_encode_shape(self, model, batch, batch_size):
        representation = model.encoder(batch)
        assert representation.shape == (batch_size, model.dictionary_dim)

    def test_decode_shape(self, model, batch):
        representation = model.encoder(batch)
        reconstruction = model.decoder(representation)
        assert reconstruction.shape == batch.shape

    def test_predict_kth_shape(self, model, batch, batch_size):
        representation = model.encoder(batch)
        for k in range(model.dictionary_dim):
            k_prediction = model.predict_kth(k, representation)
            assert k_prediction.shape == (batch_size, 1)

    def test_predict_all_shape(self, model, batch):
        representation = model.encoder(batch)
        all_predictions = model.predict_all(representation)
        assert all_predictions.shape == representation.shape