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

        def hook(module, inputs, output):
            activations = output
            encoded = self.schmidhuber.encode(activations)
            dictionary_representations.append(encoded.detach().cpu())

    def generate_with_intervention(self, generation_param: GenerationParams, dictionary_multipliers: dict[int, float]):
        pass

    def _multipliers_dict_to_tensor(self, dictionary_multipliers: dict[int, float]) -> torch.Tensor:
        pass

    def _locate_layer(self, layer_name: str) -> nn.Module:
        # TODO
        return self.diffusion.unet.up_blocks[1].attentions[2]