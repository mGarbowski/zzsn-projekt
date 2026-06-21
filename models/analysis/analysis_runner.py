import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch
import wandb
from datasets import load_dataset, Dataset, Features, Value, Sequence
from matplotlib import pyplot as plt
from torch import Tensor
from PIL.Image import Image

from models.diffusion import WrappedDiffusion, GenerationParams, GenerationResult


@dataclass
class AnalysisRunnerConfig:
    out_dir: Path

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
    intervention_strengths: list[float] = field(
        default_factory=lambda: [1.0, 0.0, -1.0, -10.0]
    )

    sample_prompt_base: str = "A picture of a british shorthair cat"


class AnalysisRunner:
    def __init__(self, cfg: AnalysisRunnerConfig) -> None:
        self.cfg = cfg

    def _load_wrapped_diffusion(self):
        return WrappedDiffusion.from_pretrained(
            schmidhuber_artifact_id=self.cfg.schmidhuber_artifact_id,
            diffusion_model_id=self.cfg.diffusion_model_id,
            device=self.cfg.device,
            safety_checker=None
        )

    def _load_prompts_dataset(self):
        return load_dataset(self.cfg.prompts_hf_repo_id, split="train")

    def _make_generation_params(self, prompts_ds: Dataset):
        prompts = list(prompts_ds["prompt"])
        return GenerationParams(
            prompts=prompts,
            num_seeds=self.cfg.num_seeds,
            num_inference_steps=self.cfg.num_inference_steps,
            guidance_scale=self.cfg.guidance_scale,
        )

    def _make_dictionary_representations_dataset(
        self, generation_results: list[GenerationResult], prompts_ds: Dataset
    ) -> Dataset:
        """Gather collected dictionary representations into hf Dataset for convenient processing."""
        prompt_to_style_mapping = dict(zip(prompts_ds["prompt"], prompts_ds["style"]))

        all_activations = [r.trajectory.tolist() for r in generation_results]
        num_timesteps = len(all_activations[0])
        dict_dim = len(all_activations[0][0])

        features = Features(
            {
                "prompt": Value("string"),
                "style": Value("string"),
                "seed": Value("int32"),
                "activations": Sequence(
                    Sequence(Value("float32"), length=dict_dim),
                    length=num_timesteps,
                ),
            }
        )
        data: dict = {
            "prompt": [r.prompt for r in generation_results],
            "style": [prompt_to_style_mapping[r.prompt] for r in generation_results],
            "seed": [r.seed for r in generation_results],
            "activations": all_activations,
        }
        return Dataset.from_dict(data, features=features)

    @staticmethod
    def split_dataset_by_style(style: str, dataset: Dataset) -> tuple[Dataset, Dataset]:
        """Split the dictionary dataset into 2 partitions.
        D_c - records for prompts with given style
        D_~c - records for prompts with all other styles
        """
        with_style = dataset.filter(lambda ex: ex["style"] == style)
        without_style = dataset.filter(lambda ex: ex["style"] != style)
        return with_style, without_style

    @staticmethod
    def compute_scores_for_style(
        style: str, dictionary_ds: Dataset, delta: float = 1e-8
    ) -> tuple[Tensor, Tensor]:
        """Compute a tensor of scores for each dimension in each timestep.

        Scores with respect to the given style and the dictionary dataset.
        scores tensor has shape (num_timesteps, dictionary_dim)

        Also calculates the mean feature activation over dataset examples in D_c partition
        for each timestep, for each dictionary_dim, tensor has shape (num_timesteps, dictionary_dim).
        this is used for scaling the intervention strength during intervention.
        """
        D_c, D_not_c = AnalysisRunner.split_dataset_by_style(style, dictionary_ds)
        assert len(D_c) > 0
        assert len(D_not_c) > 0

        # Stack activations:
        # (example, timestep, dictionary_dim)
        _to_tensor_act = lambda ex: torch.tensor(ex["activations"], dtype=torch.float32)

        activations_c = torch.stack([_to_tensor_act(ex) for ex in D_c], dim=0)
        activations_not_c = torch.stack([_to_tensor_act(ex) for ex in D_not_c], dim=0)

        # Mean over dataset examples:
        # (num_timesteps, dictionary_dim)
        mu_c = activations_c.mean(dim=0)
        mu_not_c = activations_not_c.mean(dim=0)

        # divide each score by sum over all dictionary dimensions
        norm_c = mu_c / (mu_c.sum(dim=-1, keepdim=True) + delta)
        norm_not_c = mu_not_c / (mu_not_c.sum(dim=-1, keepdim=True) + delta)

        scores = norm_c - norm_not_c
        return scores, mu_c

    @staticmethod
    def _make_histogram_of_dimension_scores(
        scores: Tensor, timesteps: list[int], style: str
    ):
        """Make a histogram of dimension scores.
        For some selected timesteps, see how many dimensions are relevant to the style (high score)
        """
        fig, ax = plt.subplots(len(timesteps))
        fig.suptitle(f"Rozkład oceny wymiarów słownika dla stylu {style}")
        for i, t in enumerate(timesteps):
            ax[i].hist(scores[t].numpy(), bins=50)
            ax[i].set_title(f"t={t}")

        plt.tight_layout()
        return fig

    @staticmethod
    def _select_top_k_features_for_style(
        scores: Tensor, mu_c: Tensor, k: int
    ) -> dict[int, float]:
        """Select top k dictionary dimensions with the highest score on average over timesteps.

        Map the dimensions indices to their average activation on the D_c partition.
        """
        aggregated_scores = torch.mean(scores, dim=0)  # average by timestep
        top_dimension_idxs = torch.topk(aggregated_scores, k=k, dim=0).indices
        mean_activations_by_dim = mu_c.mean(dim=0)
        return {
            dim_idx: mean_activations_by_dim[dim_idx].item()
            for dim_idx in top_dimension_idxs.tolist()
        }

    @staticmethod
    def analyze_style(style: str, dictionary_ds: Dataset, k: int):
        scores, mu_c = AnalysisRunner.compute_scores_for_style(style, dictionary_ds)
        histograms_plot = AnalysisRunner._make_histogram_of_dimension_scores(
            scores, [0, 10, 20, 30, 40, 50], style
        )
        top_features = AnalysisRunner._select_top_k_features_for_style(scores, mu_c, k)
        return {
            "scores": scores,
            "mu_c": mu_c,
            "histograms_plot": histograms_plot,
            "top_features": top_features,
        }

    def _make_image_row_no_intervention(
        self, diffusion: WrappedDiffusion, styles: list[str]
    ) -> list[Image]:
        params = GenerationParams(
            prompts=[
                f"{self.cfg.sample_prompt_base} in {style} style" for style in styles
            ],
            num_seeds=self.cfg.num_seeds,
            num_inference_steps=self.cfg.num_inference_steps,
            guidance_scale=self.cfg.guidance_scale,
        )
        results = diffusion.generate(params)
        return [r.image for r in results]

    def _make_image_row_with_intervention(
        self,
        diffusion: WrappedDiffusion,
        styles: list[str],
        strength: float,
        top_dimensions: dict[str, dict[int, float]],
    ) -> list[Image]:
        results = []

        for style in styles:
            params = GenerationParams(
                prompts=[f"{self.cfg.sample_prompt_base} in {style} style"],
                num_seeds=self.cfg.num_seeds,
                num_inference_steps=self.cfg.num_inference_steps,
                guidance_scale=self.cfg.guidance_scale,
            )
            dictionary_multipliers = {
                k: strength * v for k, v in top_dimensions[style].items()
            }
            result = diffusion.generate_with_intervention(
                params, dictionary_multipliers
            )
            results.append(result[0].image)

        return results

    def make_sample_images(
        self,
        diffusion: WrappedDiffusion,
        styles: list[str],
        top_dimensions: dict[str, dict[int, float]],
    ):
        no_intervention_row = self._make_image_row_no_intervention(diffusion, styles)
        intervention_rows = [
            self._make_image_row_with_intervention(
                diffusion, styles, strength, top_dimensions
            )
            for strength in self.cfg.intervention_strengths
        ]

        n_cols = len(styles)
        n_rows = 1 + len(intervention_rows)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, 6))

        for i, image in enumerate(no_intervention_row):
            axes[0, i].imshow(image)
            axes[0, i].set_title(f"{styles[i]}")
            axes[0, i].axis("off")

        for row_idx, img_row in enumerate(intervention_rows, start=1):
            for img_idx, img in enumerate(img_row):
                axes[row_idx, img_idx].imshow(img)
                axes[row_idx, img_idx].axis("off")

        row_labels = ["-"] + [str(s) for s in self.cfg.intervention_strengths]
        for r, label in enumerate(row_labels):
            ax = axes[r, 0]
            ax.set_ylabel(
                label, rotation=0, fontsize=12, va="center", ha="right", labelpad=12
            )

        fig.subplots_adjust(left=0.12, top=0.95, bottom=0.05)
        plt.tight_layout()
        return fig

    def process_and_save_results(
        self, wb_run, per_style_analysis_results, sample_images_fig
    ):
        out_dir = self.cfg.out_dir / wb_run.id
        out_dir.mkdir(parents=True, exist_ok=True)

        table = wandb.Table(columns=["style", "top_features", "score_distribution"])
        for style, result in per_style_analysis_results.items():
            style_dir = out_dir / style.replace(" ", "_")
            style_dir.mkdir(parents=True, exist_ok=True)

            scores_np = result["scores"].cpu().numpy()
            mu_c_np = result["mu_c"].cpu().numpy()
            np.save(style_dir / "scores.npy", scores_np)
            np.save(style_dir / "mu_c.npy", mu_c_np)

            top_features = result["top_features"]
            with open(style_dir / "top_features.json", "w", encoding="utf8") as f:
                json.dump(top_features, f, indent=2)

            hist_fig = result["histograms_plot"]
            hist_path = style_dir / "histogram.png"
            hist_fig.savefig(hist_path, bbox_inches="tight")
            plt.close(hist_fig)

            table.add_data(style, json.dumps(top_features), wandb.Image(str(hist_path)))

        sample_path = out_dir / "sample_images.png"
        sample_images_fig.savefig(sample_path, bbox_inches="tight")
        plt.close(sample_images_fig)

        art = wandb.Artifact(
            name=f"analysis-{wb_run.id}",
            type="analysis",
            metadata={"model_artifact": self.cfg.schmidhuber_artifact_id},
        )
        art.add_dir(str(out_dir))

        wb_run.log_artifact(art)
        wb_run.log({"analysis/summary_table": table})
        wb_run.log({"analysis/sample_images": wandb.Image(str(sample_path))})

    def run(self) -> None:
        wb_run = wandb.init(
            project=self.cfg.wandb_project,
            job_type="analysis",
            config=asdict(self.cfg),
            reinit=True,
        )
        try:
            wb_run.use_artifact(self.cfg.schmidhuber_artifact_id)

            diffusion = self._load_wrapped_diffusion()
            prompt_ds = self._load_prompts_dataset()

            generation_params = self._make_generation_params(prompt_ds)
            generation_results = diffusion.generate_and_collect_dictionary(
                generation_params, self.cfg.batch_size
            )
            dictionary_ds = self._make_dictionary_representations_dataset(
                generation_results, prompt_ds
            )

            styles = sorted(set(prompt_ds["style"]))
            per_style_analysis_results = {
                style: self.analyze_style(
                    style, dictionary_ds, self.cfg.top_k_dimensions
                )
                for style in styles
            }

            top_dimensions = {
                style: per_style_analysis_results[style]["top_features"]
                for style in styles
            }
            sample_images_fig = self.make_sample_images(
                diffusion, styles, top_dimensions
            )
            self.process_and_save_results(
                wb_run, per_style_analysis_results, sample_images_fig
            )

        finally:
            wb_run.finish()
