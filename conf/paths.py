import os
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent
HYDRA_CONFIG_ROOT = PROJECT_ROOT / "conf"
HYDRA_CONFIG_ROOT_STR = str(HYDRA_CONFIG_ROOT)


if scratch := os.getenv("SCRATCH") is not None:
    SCRATCH_ROOT = Path(os.environ["SCRATCH"])
else:
    logger.warning(
        "Failed to set $SCRATCH root. Using project_root/scratch instead. This is expected when running code outside of Athena."
    )
    SCRATCH_ROOT = PROJECT_ROOT / "scratch"
    SCRATCH_ROOT.mkdir(exist_ok=True)
