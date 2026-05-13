# List available commands
@default:
    just -l

# Code formatter
fmt:
    uvx ruff format

# Check formatting, linter and types
check:
    uvx ruff format --check
    uvx ruff check
    # uvx ty check


# Install dependencie on local machine
init-local:
  uv sync --extra cpu  # Use CPU only torch version

# Create venv in scratch space on Athena
init-athena:
  uv venv --allow-existing $SCRATCH/venvs/zzsn-projekt
  uv sync --active --extra athena  # Use CUDA capable torch version

# Execute script from scripts package (run this locally)
run-local script *ARGS: 
  uv run --active -m scripts.{{script}} -m {{ARGS}} infra=local

# Execute script from scripts package (run this on Athena)
run-athena script *ARGS:  
  source $SCRATCH/venvs/zzsn-projekt/bin/activate && UV_ENV_FILE=.env uv run --active --extra athena -m scripts.{{script}} -m {{ARGS}}

# Peek job queue on Athena
queue:
  squeue --me

# Run unit tests
test:
  uv run pytest

# Start marimo notebook editor
marimo:
  uv run marimo edit