import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_name="config", config_path="conf")
def main(_: DictConfig):
    print("hello, world")


if __name__ == "__main__":
    main()
