from dataclasses import dataclass

import torch
from diffusers import StableDiffusionPipeline
from torch import nn

from models.linear import SchmidhuberLinear


@dataclass
class GenerationParams:
    prompt: str
    num_inference_steps: int
    guidance_scale: float
    random_seed: int = 42

class WrappedDiffusion:
    """Wraps StableDiffusion 1.4 to parse the activations with trained Schmidhuber model



    Can save the dictionary representation during generation (each timestep?)
        The actiovations are passed through the encoder to producte dictionary representation
        and then saved using a hook
        The activations are processed per spatial patch (16x16)
        for each timestep we average them over the spatial (h, w) dimensions

    Can intervene in the generation
        the user provides a collection of pairs
        (dictionary space index, multiplier)
        during generation (in each timestep?)
        the activations are passed through the encoder to produce dictionary representation
        the corresponding dictionary dimensions are multiplied by the multipliers
        the representation after intervention is passed through the decoder
        the decoded activations are passed through the rest of the dffusion model
    """

    def __init__(self, diffusion: StableDiffusionPipeline, schmidhuber: SchmidhuberLinear, layer_name: str = "unet.up_blocks.1.attentions.2"):
        self.diffusion = diffusion
        self.schmidhuber = schmidhuber
        self.layer_name = layer_name

    def generate_and_collect_dictionary(self, generation_params: GenerationParams):
        dictionary_representations = []

        layer = self._locate_layer(self.layer_name)

        def hook(module, inputs, output):
            act = output
            # If output is a tuple (some modules return tuples), take the first tensor
            if isinstance(act, tuple):
                act = act[0]

            # Expected shapes:
            # - (B, C, H, W)  -> convert to (B, Npatch, C) where Npatch = H*W
            # - (B, Npatch, C) -> use as-is

            assert act.ndim == 4
            B, C, H, W = act.shape
            patches = act.view(B, C, H * W).permute(0, 2, 1)  # (B, Npatch, C)

            # Handle guidance-duplication: if batch is doubled (uncond + cond),
            # keep only conditioned half (second chunk).
            if patches.shape[0] % 2 == 0 and generation_params.guidance_scale > 1.0:
                try:
                    uncond, cond = patches.chunk(2, dim=0)
                    patches = cond
                except Exception:
                    # if chunk fails, keep patches as-is
                    pass

            B2, N, C = patches.shape
            encoder = self.schmidhuber.encoder
            device = next(encoder.parameters()).device

            # Flatten patches to (B2 * N, C) and encode in one batch
            patches_flat = patches.reshape(B2 * N, C).to(device)
            with torch.no_grad():
                encoded_flat = encoder(patches_flat)  # (B2 * N, dict_dim)

            dict_dim = encoded_flat.shape[-1]
            encoded = encoded_flat.view(B2, N, dict_dim)  # (B2, N, dict_dim)

            # Average across patches to get a single vector per batch item
            encoded_mean = encoded.mean(dim=1)  # (B2, dict_dim)

            # Store detached CPU tensor
            dictionary_representations.append(encoded_mean.detach().cpu())

        handle = layer.register_forward_hook(hook)
        try:
            # Run generation under no_grad to avoid building graphs
            with torch.no_grad():
                output = self.diffusion(
                    generation_params.prompt,
                    num_inference_steps=generation_params.num_inference_steps,
                    guidance_scale=generation_params.guidance_scale,
                )
        finally:
            handle.remove()

        # dictionary_representations is a list with one entry per hook invocation.
        # Each entry is a tensor of shape (batch_after_guidance, dict_dim) on CPU.
        return output, dictionary_representations

    def generate_with_intervention(self, generation_param: GenerationParams, dictionary_multipliers: dict[int, float]):
        multipliers = self._multipliers_dict_to_tensor(dictionary_multipliers)
        layer = self._locate_layer(self.layer_name)

        def hook(module, inputs, output):
            is_tuple = isinstance(output, tuple)
            act = output[0] if is_tuple else output

            B, C, H, W = act.shape
            with torch.no_grad():
                act_flat = act.permute(0, 2, 3, 1).reshape(B * H * W, C)
                encoded = self.schmidhuber.encoder(act_flat)
                encoded = encoded * multipliers.to(encoded.device)
                decoded = self.schmidhuber.decoder(encoded)
                modified = decoded.reshape(B, H, W, C).permute(0, 3, 1, 2)

            return (modified,) + output[1:] if is_tuple else modified

        handle = layer.register_forward_hook(hook)
        try:
            rng = torch.Generator(device=self.diffusion.device)
            rng.manual_seed(generation_param.random_seed)
            result = self.diffusion(
                prompt=generation_param.prompt,
                num_inference_steps=generation_param.num_inference_steps,
                guidance_scale=generation_param.guidance_scale,
                generator=rng
            )
        finally:
            handle.remove()

        return result

    def _multipliers_dict_to_tensor(self, dictionary_multipliers: dict[int, float]) -> torch.Tensor:
        tensor = torch.ones(self.schmidhuber.dictionary_dim)
        for idx, multiplier in dictionary_multipliers.items():
            tensor[idx] = multiplier
        return tensor

    def _locate_layer(self, layer_name: str) -> nn.Module:
        block = self.diffusion
        for step in layer_name.split("."):
            block = block[int(step)] if step.isdigit() else getattr(block, step)
        return block