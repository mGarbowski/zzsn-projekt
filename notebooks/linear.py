import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch

    return mo, torch


@app.cell
def _(torch):
    vec = torch.rand((10,))
    vec
    return (vec,)


@app.cell
def _(vec):
    k = 7
    vec[k]
    return (k,)


@app.cell
def _(k, vec):
    vec[:k]
    return


@app.cell
def _(k, vec):
    vec[k+1:]
    return


@app.cell
def _(k, torch, vec):
    without_k = torch.cat([vec[:k], vec[k+1:]])
    without_k
    return (without_k,)


@app.cell
def _(without_k):
    without_k.shape
    return


@app.cell
def _(torch):
    vec_batch = torch.rand((3, 10))
    vec_batch
    return (vec_batch,)


@app.cell
def _(k, vec_batch):
    vec_batch[:, :k]
    return


@app.cell
def _(k, vec_batch):
    vec_batch[:, k+1:]
    return


@app.cell
def _(k, torch, vec_batch):
    res_batch = torch.cat([vec_batch[:, :k], vec_batch[:, k+1:]], dim=1)
    res_batch
    return (res_batch,)


@app.cell
def _(res_batch):
    res_batch.shape
    return


@app.cell
def _(k, vec_batch):
    vec_batch[:, k]
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A tutaj z modelu
    """)
    return


@app.cell
def _():
    from models.linear import SchmidhuberLinear, SchmidhuberLinearConfig

    return SchmidhuberLinear, SchmidhuberLinearConfig


@app.cell
def _():
    import os
    os.getcwd()
    return


@app.cell
def _(SchmidhuberLinear, SchmidhuberLinearConfig):
    cfg = SchmidhuberLinearConfig(
        input_dim=16,
        expansion_factor=4,
        predictor_dropout=0.1,
        predictor_hidden_dims=[16, 16]
    )
    model = SchmidhuberLinear(cfg)
    return cfg, model


@app.cell
def _(cfg):
    cfg
    return


@app.cell
def _(model):
    model
    return


@app.cell
def _(cfg, torch):
    batch_size = 10
    batch = torch.rand((batch_size, cfg.input_dim))
    batch.shape
    return (batch,)


@app.cell
def _(batch, model):
    rep = model.encoder(batch)
    rep.shape
    return (rep,)


@app.cell
def _(model, rep):
    res = model.decoder(rep)
    res.shape
    return (res,)


@app.cell
def _(batch, res):
    res.shape == batch.shape
    return


@app.cell
def _(model, rep):
    pred = model.predict_kth(0, rep)
    pred.shape
    return


@app.cell
def _(model, rep):
    all_pred = model.predict_all(rep)
    assert rep.shape == all_pred.shape
    all_pred.shape
    return


@app.cell
def _(model, rep, torch):
    tmp = [model.predict_kth(k, rep) for k in range(model.dictionary_dim)]
    torch.cat(tmp, dim=1).shape
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
