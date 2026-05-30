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
from pathlib import Path

import hydra
from omegaconf import OmegaConf, SCMode

from activation_collection.config import register_configs
from conf.paths import HYDRA_CONFIG_ROOT_STR
from models.analysis.analysis_runner import AnalysisRunnerConfig, AnalysisRunner
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
        cfg = AnalysisRunnerConfig(
            out_dir=Path(run_cfg.out_dir),
            schmidhuber_artifact_id=run_cfg.schmidhuber_artifact_id,
            prompts_hf_repo_id=run_cfg.prompts_hf_repo_id,
            num_seeds=run_cfg.num_seeds,
            num_inference_steps=run_cfg.num_inference_steps,
            guidance_scale=run_cfg.guidance_scale,
            diffusion_model_id=run_cfg.diffusion_model_id,
            device=run_cfg.device,
            batch_size=run_cfg.batch_size,
            wandb_project=run_cfg.wandb_project,
            top_k_dimensions=run_cfg.top_k_dimensions,
            intervention_strengths=run_cfg.intervention_strengths,
            sample_prompt_base=run_cfg.sample_prompt_base,
        )
        AnalysisRunner(cfg).run()
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
