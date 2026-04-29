from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
HYDRA_CONFIG_ROOT = PROJECT_ROOT / "conf"
HYDRA_CONFIG_ROOT_STR = str(HYDRA_CONFIG_ROOT)
