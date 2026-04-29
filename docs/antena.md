Ogólne tutoriale:
- https://github.com/ensembleAI-hackathon/tasks-2026/blob/main/shared/athena.pdf
- https://docs.google.com/document/d/1YNhcTef4vTmfREBMkZnNlS6EyY8Ed8kZaAX_Fxw5rgI/edit?tab=t.0

## Dostęp do repo

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
