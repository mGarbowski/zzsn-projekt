import hydra
from omegaconf import DictConfig
import torch
from diffusers import StableDiffusionPipeline
import os


@hydra.main(version_base=None, config_name="config", config_path="conf")
def main(_: DictConfig):
    print("hello, world")
    print(os.getcwd())
    print(torch.cuda.is_available())


    model_id = "CompVis/stable-diffusion-v1-4"
    device = "cuda"


    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)

    prompt = "capybara riding a unicycle"
    image = pipe(prompt).images[0]  
        
    image.save("capybara_rides_unicycle.png")




if __name__ == "__main__":
    main()
