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
    CacheActivationsRunner(runner_cfg).load_and_push_to_hub()


if __name__ == "__main__":
    main()
