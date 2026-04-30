init-athena:
  uv venv $SCRATCH/venvs/zzsn-projekt

venv-athena:
  source $SCRATCH/venvs/zzsn-projekt/bin/activate

run-athena *ARGS:  
  UV_ENV_FILE=.env uv run --active -m main -m {{ARGS}}
