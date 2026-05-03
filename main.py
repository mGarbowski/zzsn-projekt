import os
import re
from pathlib import Path

import hydra
import torch
from diffusers import StableDiffusionPipeline
from omegaconf import DictConfig


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


@hydra.main(version_base=None, config_name="config", config_path="conf")
def main(_: DictConfig):
    print("hello, world")
    print(os.getcwd())
    print(torch.cuda.is_available())

    model_id = "CompVis/stable-diffusion-v1-4"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    target_layer = pipe.unet.up_blocks[1].attentions[2]

    prompts = [
        "capybara riding a unicycle",
        "a photo of a cat wearing a hat",
        "a painting of a futuristic cityscape at sunset",
        "a surreal landscape with floating islands and waterfalls",
        "a portrait of a person with a galaxy for a face",
    ]

    num_inference_steps = 30
    guidance_scale = 7.5

    output_dir = Path("activations")
    images_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    activations_path = output_dir / "up_blocks_1_attentions_2.pt"

    all_results = {}

    for prompt_index, prompt in enumerate(prompts):
        print(f"Processing prompt: {prompt}")
        generator = torch.Generator(device=device).manual_seed(0)

        prompt_embeds = pipe._encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=guidance_scale > 1.0,
            negative_prompt=None,
        )

        pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        latents = pipe.prepare_latents(
            batch_size=1,
            num_channels_latents=pipe.unet.config.in_channels,
            height=pipe.unet.config.sample_size * pipe.vae_scale_factor,
            width=pipe.unet.config.sample_size * pipe.vae_scale_factor,
            dtype=dtype,
            device=device,
            generator=generator,
        )

        prompt_records = []
        step_state = {"activation": None}

        def hook_fn(_, __, output):
            if isinstance(output, tuple):
                output = output[0]
            step_state["activation"] = output.detach().to("cpu")

        hook_handle = target_layer.register_forward_hook(hook_fn)

        try:
            for step_index, timestep in enumerate(pipe.scheduler.timesteps):
                step_state["activation"] = None

                latent_model_input = latents
                if guidance_scale > 1.0:
                    latent_model_input = torch.cat([latents, latents], dim=0)

                latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timestep)

                with torch.no_grad():
                    noise_pred = pipe.unet(
                        latent_model_input,
                        timestep,
                        encoder_hidden_states=prompt_embeds,
                    ).sample

                if guidance_scale > 1.0:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                latents = pipe.scheduler.step(noise_pred, timestep, latents).prev_sample

                if step_state["activation"] is None:
                    raise RuntimeError(
                        f"Did not capture an activation at step {step_index} for prompt: {prompt}"
                    )

                prompt_records.append(
                    {
                        "step_index": step_index,
                        "timestep": int(timestep.item()) if hasattr(timestep, "item") else int(timestep),
                        "activation": step_state["activation"],
                    }
                )
        finally:
            hook_handle.remove()

        with torch.no_grad():
            image_tensor = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
            image = pipe.image_processor.postprocess(image_tensor, output_type="pil")[0]

        image_path = images_dir / f"{prompt_index:02d}_{slugify(prompt)}.png"
        image.save(image_path)
        print(f"Saved image to {image_path}")

        all_results[prompt] = prompt_records

    torch.save(
        {
            "model_id": model_id,
            "layer": "pipe.unet.up_blocks[1].attentions[2]",
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "prompts": prompts,
            "results": all_results,
        },
        activations_path,
    )

    print(f"Saved activations to {activations_path}")


if __name__ == "__main__":
    main()