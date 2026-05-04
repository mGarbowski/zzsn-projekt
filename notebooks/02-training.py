import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch
    from datasets import Dataset, load_dataset

    from models.training import TrainerConfig, Trainer
    from models.linear import SchmidhuberLinear, SchmidhuberLinearConfig

    return (
        Dataset,
        SchmidhuberLinear,
        SchmidhuberLinearConfig,
        Trainer,
        TrainerConfig,
        load_dataset,
        mo,
        torch,
    )


@app.cell
def _(SchmidhuberLinearConfig, TrainerConfig):
    model_config = SchmidhuberLinearConfig(
        input_dim=1280,
        expansion_factor=2,
        predictor_hidden_dims=[256],
        predictor_dropout=0.3,
        predictor_embedding_dim=128,
    )

    trainer_config = TrainerConfig(
        model_config=model_config,
        batch_size=32,
        num_epochs=3,
        learning_rate_predictors=4e-4,
        learning_rate_autoencoder=1e-4,
        reconstruction_loss_weight=1.0,
        dataset_repo_id="mgarbowski/zzsn-activations-1",
    )
    return model_config, trainer_config


@app.cell
def _(SchmidhuberLinear, model_config):
    model = SchmidhuberLinear(config=model_config)
    model.num_parameters()
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Spłaszczeneie zbioru danych
    * Przy zbieraniu aktywacji dla każdego obrazka mamy (1280, 16, 16) -> (256, 1280)
      * 256 tokenów przestrzennych
    * Tutaj przetwarzamy każdy token oddzielnie, chcemy batchować je po mniej niż 256 na raz
    """)
    return


@app.cell
def _(Dataset, load_dataset, trainer_config):
    data = load_dataset(trainer_config.dataset_repo_id)
    data = data.with_format("pytorch")

    flat_activations = []
    flat_timesteps = []
    flat_prompts = []

    for row in data["train"]:
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
    return (flat_train,)


@app.cell
def _(flat_train, torch, trainer_config):
    loader = torch.utils.data.DataLoader(
        flat_train, batch_size=trainer_config.batch_size
    )
    batch = next(iter(loader))
    batch
    return batch, loader


@app.cell
def _(batch):
    xs = batch["activations"]
    xs.shape
    return


@app.cell
def _(Trainer, model, trainer_config):
    trainer = Trainer(trainer_config, model)
    return (trainer,)


@app.cell
def _(loader, trainer):
    _ = trainer.train(loader)
    return


@app.cell
def _(loader):
    len(loader)
    return


if __name__ == "__main__":
    app.run()
