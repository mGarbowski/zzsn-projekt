"""Script for analyzing a trained autoencoder

Takes trained model weights from wandb artifact
Collects the dictionary representations on a dataset of prompts concerning different styles
Calculate per dimension scores for each concept (style) in the prompt dataset
Select the top k dimensions and calculate their average activations
Generate few example images with and without intervention

Saves the results as wandb artifacts
"""
import sys
import traceback

import hydra
from omegaconf import OmegaConf, SCMode

from activation_collection.config import register_configs
from conf.paths import HYDRA_CONFIG_ROOT_STR
from models.analysis.analysis_script_config import AnalyzeAutoencoderScriptConfig

register_configs()


@hydra.main(
    version_base=None,
    config_name="analyze_autoencoder",
    config_path=HYDRA_CONFIG_ROOT_STR,
)
def main(cfg: AnalyzeAutoencoderScriptConfig) -> None:
    run_cfg: AnalyzeAutoencoderScriptConfig = OmegaConf.to_container(
        cfg, resolve=True, structured_config_mode=SCMode.INSTANTIATE
    )
    try:
        pass
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == '__main__':
    main()
