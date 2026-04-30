from dataclasses import dataclass
from typing import List, Optional

from hydra.core.config_store import ConfigStore

from conf.paths import SCRATCH_ROOT


@dataclass
class CacheActivationsRunnerConfig:
    hook_names: Optional[List[str]] = None
    new_cached_activations_path: Optional[str] = None
    dataset_name: str = "guangyil/laion-coco-aesthetic"
    split: str = "train"
    column: str = "caption"
    device: str = "cuda"
    model_name: str = "CompVis/stable-diffusion-v1-4"
    dtype: str = "float16"
    num_inference_steps: int = 50
    seed: int = 42
    batch_size_per_gpu: int = 100
    num_workers: int = 8
    output_or_diff: str = "output"
    max_num_examples: Optional[int] = None
    cache_every_n_timesteps: int = 1
    guidance_scale: float = 7.5

    hf_repo_id: Optional[str] = None
    hf_num_shards: Optional[int] = None
    hf_revision: str = "main"
    hf_is_private_repo: bool = False

    def __post_init__(self):
        if self.new_cached_activations_path is None:
            self.new_cached_activations_path = str(
                SCRATCH_ROOT
                / "activations"
                / f"{self.dataset_name.split('/')[-1]}"
                / f"{self.model_name.split('/')[-1]}"
                / f"{self.output_or_diff}"
            )
        if isinstance(self.hook_names, str):
            self.hook_names = [self.hook_names]


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="collect_activations_config", node=CacheActivationsRunnerConfig)
