import hydra
from omegaconf import DictConfig
import torch
from diffusers import AutoPipelineForText2Image
import os


@hydra.main(version_base=None, config_name="config", config_path="conf")
def main(_: DictConfig):
    print("hello, world")
    print(os.getcwd())
    print(torch.cuda.is_available())


    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
    pipe.to("cuda")

    prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."

    image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
    image.save("output.png")



if __name__ == "__main__":
    main()
