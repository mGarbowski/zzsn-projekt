from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset, Features, Image, Sequence, Value

from models.diffusion import GenerationParams, GenerationResult, WrappedDiffusion


@dataclass
class DictionaryCollectionConfig:
    generation_params: GenerationParams
    output_dir: str | Path
    batch_size: int = 1
    save_images: bool = False
    hf_repo_id: str | None = None
    hf_private: bool = False


class DictionaryActivationsRunner:
    """Collects Schmidhuber dictionary representations across prompts and seeds.

    For every (prompt, seed) pair in generation_params, runs one full diffusion
    generation and records the per-timestep dictionary representation (spatial mean
    over patches) at the configured layer.  Results are saved as a HuggingFace
    Dataset where each row corresponds to one (prompt, seed) pair:

        prompt      – str
        seed        – int
        activations – float32 array of shape (num_timesteps, dict_dim)
        image       – PIL image (only when cfg.save_images=True)
    """

    def __init__(self, cfg: DictionaryCollectionConfig, wrapped: WrappedDiffusion):
        self.cfg = cfg
        self.wrapped = wrapped

    def run(self) -> Dataset:
        cfg = self.cfg

        results: list[GenerationResult] = self.wrapped.generate_and_collect_dictionary(
            cfg.generation_params,
            batch_size=cfg.batch_size,
        )

        all_activations = [r.trajectory.tolist() for r in results]
        num_timesteps = len(all_activations[0])
        dict_dim = len(all_activations[0][0])

        features = Features(
            {
                "prompt": Value("string"),
                "seed": Value("int32"),
                "activations": Sequence(
                    Sequence(Value("float32"), length=dict_dim),
                    length=num_timesteps,
                ),
            }
        )

        data: dict = {
            "prompt": [r.prompt for r in results],
            "seed": [r.seed for r in results],
            "activations": all_activations,
        }

        if cfg.save_images:
            features["image"] = Image()
            data["image"] = [r.image for r in results]

        dataset = Dataset.from_dict(data, features=features)

        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(output_dir))

        if cfg.hf_repo_id:
            dataset.push_to_hub(cfg.hf_repo_id, private=cfg.hf_private)

        return dataset
