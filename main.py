import hydra
from omegaconf import DictConfig
import torch


@hydra.main(version_base=None, config_name="config", config_path="conf")
def main(_: DictConfig):
    print("hello, world")
    print(torch.cuda.is_available())


if __name__ == "__main__":
    main()
