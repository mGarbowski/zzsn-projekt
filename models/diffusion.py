"""Wrapper for Stable Diffusion to use with the Schmidhuber model.
To intervene in the generation and/or collect the dictionary representations for analysis
"""

from dataclasses import dataclass
from itertools import islice

import torch
from diffusers import StableDiffusionPipeline
from PIL.Image import Image
from torch import nn

from models.linear import SchmidhuberLinear


@dataclass
class GenerationParams:
    prompts: list[str]
    num_seeds: int
    """seeds 0..num_seeds-1 are tried for every prompt"""

    num_inference_steps: int
    guidance_scale: float


@dataclass
class GenerationResult:
    prompt: str
    seed: int
    image: Image
    trajectory: torch.Tensor | None = None
    """(num_timesteps, dict_dim); None for interventions"""


def _chunked(iterable, n: int):
    it = iter(iterable)
    while chunk := list(islice(it, n)):
        yield chunk


class WrappedDiffusion:
    """Wraps StableDiffusion 1.4 to parse the activations with trained Schmidhuber model

    Can save the dictionary representation during generation (in each timestep)
        The actiovations are passed through the encoder to producte dictionary representation
        and then saved using a hook
        The activations are processed per spatial patch (16x16)
        for each timestep we average them over the spatial (h, w) dimensions

    Can intervene in the generation
        the user provides a collection of pairs
        (dictionary space index, multiplier)
        during generation (in each timestep)
        the activations are passed through the encoder to produce dictionary representation
        the corresponding dictionary dimensions are multiplied by the multipliers
        the representation after intervention is passed through the decoder
        the decoded activations are passed through the rest of the dffusion model
    """

    def __init__(
        self,
        diffusion: StableDiffusionPipeline,
        schmidhuber: SchmidhuberLinear,
        layer_name: str = "unet.up_blocks.1.attentions.2",
    ):
        self.diffusion = diffusion
        self.schmidhuber = schmidhuber
        self.layer_name = layer_name

    @classmethod
    def from_pretrained(
        cls,
        schmidhuber_artifact_id: str,
        diffusion_model_id: str = "CompVis/stable-diffusion-v1-4",
        device: str = "cpu",
        **diffusion_kwargs,
    ) -> "WrappedDiffusion":
        """Load a WrappedDiffusion from a pretrained diffusion model and a W&B artifact.

        Args:
            schmidhuber_artifact_id: W&B artifact identifier for the Schmidhuber checkpoint, e.g. "entity/project/model-{run_id}-epoch_9:latest".
            diffusion_model_id: HuggingFace model ID passed to StableDiffusionPipeline.from_pretrained

            **diffusion_kwargs: forwarded to StableDiffusionPipeline.from_pretrained.
        """
        diffusion = StableDiffusionPipeline.from_pretrained(
            diffusion_model_id, **diffusion_kwargs
        ).to(device)
        schmidhuber = SchmidhuberLinear.from_wandb_artifact(
            schmidhuber_artifact_id, device=device
        )
        return cls(diffusion, schmidhuber, layer_name=schmidhuber.cfg.layer_name)

    def generate_and_collect_dictionary(
        self,
        params: GenerationParams,
        batch_size: int = 1,
    ) -> list[GenerationResult]:
        """Run generation for all (prompt, seed) pairs and collect dictionary representations.

        Iterates the Cartesian product of params.prompts × range(params.num_seeds)
        in batches of batch_size.

        Returns one GenerationResult per (prompt, seed) pair, each carrying the
        prompt, seed, generated image, and activation trajectory of shape (num_timesteps, dict_dim).
        """
        pairs = [
            (prompt, seed)
            for prompt in params.prompts
            for seed in range(params.num_seeds)
        ]

        results: list[GenerationResult] = []

        layer = self._locate_layer(self.layer_name)

        for batch in _chunked(pairs, batch_size):
            batch_prompts = [p for p, _ in batch]
            batch_seeds = [s for _, s in batch]
            generators = [
                torch.Generator(device="cpu").manual_seed(s) for s in batch_seeds
            ]

            per_timestep: list[torch.Tensor] = []

            # Default-arg trick: captures the current list object at definition time,
            # avoiding stale-closure issues inside the for loop.
            def hook(_module, _inputs, output, _buf=per_timestep):
                act = output[0] if isinstance(output, tuple) else output

                assert act.ndim == 4
                B, C, H, W = act.shape
                patches = act.view(B, C, H * W).permute(0, 2, 1)  # (B, N, C)

                # With CFG the UNet doubles the batch (uncond + cond); keep cond half.
                if patches.shape[0] % 2 == 0 and params.guidance_scale > 1.0:
                    _, patches = patches.chunk(2, dim=0)

                B2, N, C2 = patches.shape
                encoder = self.schmidhuber.encoder
                param = next(encoder.parameters())

                patches_flat = patches.reshape(B2 * N, C2).to(
                    device=param.device, dtype=param.dtype
                )
                with torch.no_grad():
                    encoded_flat = encoder(patches_flat)  # (B2*N, dict_dim)

                dict_dim = encoded_flat.shape[-1]
                encoded_mean = encoded_flat.view(B2, N, dict_dim).mean(
                    dim=1
                )  # (B2, dict_dim)
                _buf.append(encoded_mean.detach().cpu())

            handle = layer.register_forward_hook(hook)
            try:
                with torch.no_grad():
                    output = self.diffusion(
                        batch_prompts,
                        num_inference_steps=params.num_inference_steps,
                        guidance_scale=params.guidance_scale,
                        generator=generators,
                    )
            finally:
                handle.remove()

            # per_timestep: T x (B, dict_dim) -> (T, B, dict_dim) -> B x (T, dict_dim)
            stacked = torch.stack(per_timestep, dim=0)
            for i, (prompt, seed) in enumerate(batch):
                results.append(
                    GenerationResult(
                        prompt=prompt,
                        seed=seed,
                        image=output.images[i],
                        trajectory=stacked[:, i, :],
                    )
                )

        return results

    def generate_and_collect_heatmaps(
        self,
        params: GenerationParams,
        features_to_collect: list[int],
        batch_size: int = 1,
    ) -> list[tuple[GenerationResult, torch.Tensor]]:
        """Run generation for all (prompt, seed) pairs and collect activation heatmaps.

        Iterates the Cartesian product of params.prompts x range(params.num_seeds)
        in batches of batch_size.

        Returns one (GenerationResult, heatmap tensor) pair per (prompt, seed) pair.
        Heatmap is a tensor of shape (n_features, n_timesteps, H, W)
        containing the per spatial patch activations of the specified features across the generation trajectory.
        """
        if not features_to_collect:
            raise ValueError("features_to_collect must not be empty")

        pairs = [
            (prompt, seed)
            for prompt in params.prompts
            for seed in range(params.num_seeds)
        ]

        results: list[tuple[GenerationResult, torch.Tensor]] = []

        layer = self._locate_layer(self.layer_name)

        for batch in _chunked(pairs, batch_size):
            batch_prompts = [p for p, _ in batch]
            batch_seeds = [s for _, s in batch]
            generators = [
                torch.Generator(device="cpu").manual_seed(s) for s in batch_seeds
            ]

            per_timestep: list[torch.Tensor] = []

            def hook(_module, _inputs, output, _buf=per_timestep):
                act = output[0] if isinstance(output, tuple) else output

                assert act.ndim == 4
                B, C, H, W = act.shape
                patches = act.view(B, C, H * W).permute(0, 2, 1)  # (B, N, C)

                # With CFG the UNet doubles the batch (uncond + cond); keep cond half.
                if patches.shape[0] % 2 == 0 and params.guidance_scale > 1.0:
                    _, patches = patches.chunk(2, dim=0)

                B2, N, C2 = patches.shape
                encoder = self.schmidhuber.encoder
                param = next(encoder.parameters())

                patches_flat = patches.reshape(B2 * N, C2).to(
                    device=param.device, dtype=param.dtype
                )
                with torch.no_grad():
                    encoded_flat = encoder(patches_flat)  # (B2*N, dict_dim)

                encoded = encoded_flat.view(B2, N, -1)  # (B2, N, dict_dim)
                encoded = encoded[
                    :, :, features_to_collect
                ]  # only the selected dimensions from dict_dim
                encoded = encoded.permute(0, 2, 1).reshape(
                    B2, len(features_to_collect), H, W
                )
                _buf.append(encoded.detach().cpu())

            handle = layer.register_forward_hook(hook)
            try:
                with torch.no_grad():
                    output = self.diffusion(
                        batch_prompts,
                        num_inference_steps=params.num_inference_steps,
                        guidance_scale=params.guidance_scale,
                        generator=generators,
                    )
            finally:
                handle.remove()

            stacked = torch.stack(per_timestep, dim=0)  # (T, B, F, H, W)
            for i, (prompt, seed) in enumerate(batch):
                results.append(
                    (
                        GenerationResult(
                            prompt=prompt,
                            seed=seed,
                            image=output.images[i],
                        ),
                        stacked[:, i, :, :, :].permute(1, 0, 2, 3).contiguous(),
                    )
                )

        return results

    def generate(
        self,
        params: GenerationParams,
        batch_size: int = 1,
    ) -> list[GenerationResult]:
        """Run plain generation for all (prompt, seed) pairs with no hooks.

        Returns one GenerationResult per (prompt, seed) pair (trajectory is None).
        """
        pairs = [
            (prompt, seed)
            for prompt in params.prompts
            for seed in range(params.num_seeds)
        ]

        results: list[GenerationResult] = []

        for batch in _chunked(pairs, batch_size):
            batch_prompts = [p for p, _ in batch]
            batch_seeds = [s for _, s in batch]
            generators = [
                torch.Generator(device="cpu").manual_seed(s) for s in batch_seeds
            ]
            with torch.no_grad():
                output = self.diffusion(
                    batch_prompts,
                    num_inference_steps=params.num_inference_steps,
                    guidance_scale=params.guidance_scale,
                    generator=generators,
                )
            for i, (prompt, seed) in enumerate(batch):
                results.append(
                    GenerationResult(prompt=prompt, seed=seed, image=output.images[i])
                )

        return results

    def generate_with_intervention(
        self,
        params: GenerationParams,
        dictionary_multipliers: dict[int, float],
        batch_size: int = 1,
    ) -> list[GenerationResult]:
        """Run generation with dictionary-space intervention for all (prompt, seed) pairs.

        Compensate the reconstruction errors by first running the activations through th Schmidhuber model
        without intervention, calculating the difference original-reconstruction.
        Then adding this difference to the reconstructed activations after intervention.

        Returns one GenerationResult per (prompt, seed) pair. trajectory is None
        since activations are not collected during intervention.
        """
        multipliers = self._multipliers_dict_to_tensor(dictionary_multipliers)

        layer = self._locate_layer(self.layer_name)

        def hook(_module, _inputs, output):
            is_tuple = isinstance(output, tuple)
            act = output[0] if is_tuple else output

            B, C, H, W = act.shape
            with torch.no_grad():
                param = next(self.schmidhuber.parameters())
                act_flat = act.permute(0, 2, 3, 1).reshape(B * H * W, C)
                act_flat = act_flat.to(device=param.device, dtype=param.dtype)

                encoded = self.schmidhuber.encoder(act_flat)

                # Pass without intervention to calculate the reconstruction diff
                decoded_no_intervention = self.schmidhuber.decoder(encoded)
                reconstruction_diff = act_flat - decoded_no_intervention

                # Pass with intervention
                latent_with_intervention = encoded * multipliers.to(
                    device=encoded.device, dtype=encoded.dtype
                )
                decoded = self.schmidhuber.decoder(latent_with_intervention)
                decoded = (
                    decoded + reconstruction_diff
                )  # compensate for the reconstruction errors
                modified = decoded.reshape(B, H, W, C).permute(0, 3, 1, 2)
                modified = modified.to(device=act.device, dtype=act.dtype)

            return (modified,) + output[1:] if is_tuple else modified

        handle = layer.register_forward_hook(hook)
        try:
            results = self.generate(params, batch_size)
        finally:
            handle.remove()

        return results

    def _multipliers_dict_to_tensor(
        self, dictionary_multipliers: dict[int, float]
    ) -> torch.Tensor:
        """Convert a dictionary of multipliers to a tensor.

        {1: 11, 3: 33} -> [1, 11, 1, 33, 1, 1, 1...]
        """
        param = next(self.schmidhuber.parameters())
        tensor = torch.ones(
            self.schmidhuber.dictionary_dim,
            device=param.device,
            dtype=param.dtype,
        )
        for idx, multiplier in dictionary_multipliers.items():
            tensor[idx] = multiplier
        return tensor

    def _locate_layer(self, layer_name: str) -> nn.Module:
        """Select a layer from the diffusion model based on a dotted path.
        Like "unet.up_blocks.1.attentions.2"
        """
        block = self.diffusion
        for step in layer_name.split("."):
            block = block[int(step)] if step.isdigit() else getattr(block, step)
        return block
