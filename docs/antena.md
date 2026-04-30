Ogólne tutoriale:
- https://github.com/ensembleAI-hackathon/tasks-2026/blob/main/shared/athena.pdf
- https://docs.google.com/document/d/1YNhcTef4vTmfREBMkZnNlS6EyY8Ed8kZaAX_Fxw5rgI/edit?tab=t.0

## Dostęp do repo

0. Aktywuj dostęp do Atheny na [stronce](https://portal.plgrid.pl/)

1. Z poziomu anteny `ssh-keygen` i zrzucasz klucz np. do `~/.ssh/id_zzsn_github`:

2. `~/.ssh/config`:
```

Host github.com
        User git
        IdentityFile ~/.ssh/id_zzsn_github

```

3. Dodajesz deploy key przez settings w repo githubowym

4. Klonujesz i powinno śmigać

## Setup uv + cache

1. Instalacja:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Ważne** - wyrzucenie cache do `$SCRATCH`

```bash
echo "export UV_CACHE_DIR=$SCRATCH/uv_cache" >> ~/.bashrc
```

3. To samo dla HF:
```bash
echo "export HF_HUB_CACHE=$SCRATCH/hf_hub_cache" >> ~/.bashrc
echo "export HF_HOME=$SCRATCH/hf_home" >> ~/.bashrc
```

4. Tworzenie venva

https://pydevtools.com/handbook/how-to/how-to-customize-uvs-virtual-environment-location/

```bash
uv venv $SCRATCH/venvs/zzsn-projekt # create venv in scratch
source $SCRATCH/venvs/zzsn-projekt/bin/activate # activate it
uv sync --active # install deps
uv run --active ....  # use it to run code
```

Alernatywnie po zainstalowaniu justa (`curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to $HOME/bin`):
```bash
just init-athena
just venv-athena
just run-athena [args]
```

## Uruchamianie kodu lokalnie i na athenie

- https://hydra.cc/docs/plugins/submitit_launcher/

Generalnie, żeby odpalić job przez slurma, trzeba uruchomić apkę hydrową z flagą -m/--multirun, np. `uv run -m main -m <opcje do nadpisania>`.

Odpalanie bez -m uruchamia z domyślnym launcherem, można tak odpalać lokalnie pojedynczy config. Żeby lokalnie odpalić multirun trzeba nadpisać opcję `infra`: `uv run -m main -m infra=local`.

Żeby uruchamiać joby na athenie, potrzeba wypełnionego enva na wzór `.env.example` i ustawienia `UV_ENV_FILE` (może być przez direnv i `.envrc`, ale trzeba pobrać curlem).
