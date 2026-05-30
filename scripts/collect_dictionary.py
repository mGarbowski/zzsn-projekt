# scripts/collect_dictionary.py
# example usage:
#   python scripts/collect_dictionary.py \
#     schmidhuber_artifact_id=entity/project/model-abc123-epoch_9:latest \
#     num_seeds=3 \
#     batch_size=4

import sys
import traceback

import hydra
import torch
from datasets import load_dataset
from omegaconf import OmegaConf, SCMode

from activation_collection.dictionary_config import (
    DictionaryCollectionScriptConfig,
    register_configs,
)
from activation_collection.dictionary_runner import (
    DictionaryActivationsRunner,
    DictionaryCollectionConfig,
)
from conf.paths import HYDRA_CONFIG_ROOT_STR
from models.diffusion import GenerationParams, WrappedDiffusion

register_configs()


@hydra.main(
    version_base=None,
    config_name="collect_dictionary",
    config_path=HYDRA_CONFIG_ROOT_STR,
)
def main(cfg: DictionaryCollectionScriptConfig) -> None:
    run_cfg: DictionaryCollectionScriptConfig = OmegaConf.to_container(
        cfg, resolve=True, structured_config_mode=SCMode.INSTANTIATE
    )
    try:
        prompts_dataset = load_dataset(run_cfg.prompts_hf_repo_id)
        prompts = list(prompts_dataset["train"]["prompt"])

        wrapped = WrappedDiffusion.from_pretrained(
            schmidhuber_artifact_id=run_cfg.schmidhuber_artifact_id,
            diffusion_model_id=run_cfg.diffusion_model_id,
            device=run_cfg.device,
            torch_dtype=getattr(torch, run_cfg.dtype),
        )

        generation_params = GenerationParams(
            prompts=prompts,
            num_seeds=run_cfg.num_seeds,
            num_inference_steps=run_cfg.num_inference_steps,
            guidance_scale=run_cfg.guidance_scale,
        )

        runner_cfg = DictionaryCollectionConfig(
            generation_params=generation_params,
            output_dir=run_cfg.output_dir,
            batch_size=run_cfg.batch_size,
            save_images=run_cfg.save_images,
            hf_repo_id=run_cfg.hf_repo_id,
            hf_private=run_cfg.hf_private,
        )

        DictionaryActivationsRunner(runner_cfg, wrapped).run()

    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
