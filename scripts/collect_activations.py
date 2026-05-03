# example toy usage: just run-athena collect_activations num_workers=1 max_num_examples=1 hook_names=[unet.up_blocks.1.attentions.1]

import sys
import traceback

import hydra
from omegaconf import OmegaConf, SCMode

from activation_collection.cache_activations_runner import CacheActivationsRunner
from activation_collection.config import CacheActivationsRunnerConfig, register_configs
from conf.paths import HYDRA_CONFIG_ROOT_STR

register_configs()


@hydra.main(
    version_base=None,
    config_name="collect_activations",
    config_path=HYDRA_CONFIG_ROOT_STR,
)
def main(cfg: CacheActivationsRunnerConfig) -> None:
    # have to instantiate instead of duck typing so that the new_cached_activations_path gets computed
    runner_cfg: CacheActivationsRunnerConfig = OmegaConf.to_container(  # ty: ignore[invalid-assignment]
        cfg, resolve=True, structured_config_mode=SCMode.INSTANTIATE
    )
    try:
        CacheActivationsRunner(runner_cfg).run()
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
