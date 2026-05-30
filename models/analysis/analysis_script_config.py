from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore


@dataclass
class AnalyzeAutoencoderScriptConfig:
    out_dir: str

    schmidhuber_artifact_id: str
    prompts_hf_repo_id: str = "mgarbowski/zzsn-style-prompts"

    # Image generation params
    num_seeds: int = 1
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    diffusion_model_id: str = "CompVis/stable-diffusion-v1-4"
    device: str = "cuda"
    batch_size: int = 1

    wandb_project: str = "zzsn-projekt"

    top_k_dimensions: int = 10
    intervention_strengths: list[int] = field(default_factory=lambda: [1, 0, -1, -10])

    sample_prompt_base: str = "A picture of a british shorthair cat"


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="analyze_autoencoder_config", node=AnalyzeAutoencoderScriptConfig)