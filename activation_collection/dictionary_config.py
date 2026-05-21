from dataclasses import dataclass
from typing import Optional

from hydra.core.config_store import ConfigStore


@dataclass
class DictionaryCollectionScriptConfig:
    # W&B artifact id for the Schmidhuber checkpoint
    schmidhuber_artifact_id: str
    output_dir: str

    prompts: list[str]

    num_seeds: int = 1
    num_inference_steps: int = 50
    guidance_scale: float = 7.5

    diffusion_model_id: str = "CompVis/stable-diffusion-v1-4"
    dtype: str = "float16"

    device: str = "cuda"
    batch_size: int = 1

    save_images: bool = False
    hf_repo_id: Optional[str] = None
    hf_private: bool = False


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="dictionary_collection_config", node=DictionaryCollectionScriptConfig)
