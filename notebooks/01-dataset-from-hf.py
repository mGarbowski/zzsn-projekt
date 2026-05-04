import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import datasets

    return datasets, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Koncept

    * Na Athenie uruchamiamy skrypt, który zbiera aktywacje modelu dyfuzyjnego dla różnych promptów dla danej warsywy modelu dyfuzyjnego
    * Skrypt tworzy zbiór danych
      * dla jednej warstwy
      * element zbioru - dla każdego kroku odszumiania dla każdego prompta
    * Zbiór danych jest w formacie biblioteki `datasets`, wrzucamy go na huggingface hub

    Wstępny wynik przetwarzania (dla 10 promptów ze zbioru Coco) jest [tutaj](https://huggingface.co/datasets/mgarbowski/zzsn-activations-1)

    (na potrzeby developmentu)
    """)
    return


@app.cell
def _(datasets):
    repo_id = "mgarbowski/zzsn-activations-1"
    data = datasets.load_dataset(repo_id)
    data = data.with_format("torch")  # atrybuty ładują się jako tensory z torcha
    return (data,)


@app.cell
def _(data):
    data
    return


@app.cell
def _(data):
    data["train"]
    return


@app.cell
def _(data):
    element = next(iter(data["train"]))
    element.keys()
    return (element,)


@app.cell
def _(element):
    element["activations"]
    return


@app.cell
def _(element):
    element["activations"].shape
    return


@app.cell
def _(element):
    element["timestep"]
    return


@app.cell
def _(element):
    element["prompt"]
    return


@app.cell
def _(element):
    print(type(element["activations"]))
    print(type(element["timestep"]))
    print(type(element["prompt"]))
    return


@app.cell
def _():
    return


@app.cell
def _(data):
    from datasets import Dataset

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
def _():
    return


@app.cell
def _(flat_train):
    element_flat = next(iter(flat_train))
    print(element_flat["activations"].shape)  # torch.Size([1280])
    print(type(element_flat["activations"]))
    return


@app.cell
def _():
    return


@app.cell
def _(flat_train):
    from torch.utils.data import DataLoader

    loader = DataLoader(flat_train, batch_size=32, shuffle=True)
    batch = next(iter(loader))
    print(batch["activations"].shape)  # torch.Size([32, 1280])
    return


if __name__ == "__main__":
    app.run()
