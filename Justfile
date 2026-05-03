init-athena:
  uv venv $SCRATCH/venvs/zzsn-projekt


run-local script *ARGS: 
  uv run --active -m scripts.{{script}} -m {{ARGS}} infra=local

run-athena script *ARGS:  
  source $SCRATCH/venvs/zzsn-projekt/bin/activate && UV_ENV_FILE=.env uv run --active -m scripts.{{script}} -m {{ARGS}}

test:
  uv run pytest