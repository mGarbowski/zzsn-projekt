import io
import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path

import torch
from accelerate import Accelerator
from datasets import Array2D, Dataset, Features, Value, load_dataset
from datasets.fingerprint import generate_fingerprint
from huggingface_hub import HfApi
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from activation_collection.config import CacheActivationsRunnerConfig

torch.backends.cuda.matmul.allow_tf32 = True
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True

TORCH_STRING_DTYPE_MAP = {
    torch.float16: "float16",
    torch.float32: "float32",
    torch.bfloat16: "bfloat16",
}

DTYPE_MAP = {v: k for k, v in TORCH_STRING_DTYPE_MAP.items()}


def _cfg_to_json(cfg: CacheActivationsRunnerConfig) -> str:
    d = asdict(cfg)
    d["dtype"] = TORCH_STRING_DTYPE_MAP.get(cfg.dtype, str(cfg.dtype))
    d["new_cached_activations_path"] = str(cfg.new_cached_activations_path)
    return json.dumps(d, indent=2, ensure_ascii=False)


class CacheActivationsRunner:
    def __init__(self, cfg: CacheActivationsRunnerConfig):
        self.cfg = cfg
        if isinstance(self.cfg.dtype, str):
            self.cfg.dtype = DTYPE_MAP[self.cfg.dtype]
        self.accelerator = Accelerator()

        if self.cfg.hook_names is not None:
            if (
                "xl" in self.cfg.model_name.lower()
                or "sdxl" in self.cfg.model_name.lower()
            ):
                from activation_collection.hooked_sd14_pipeline import (
                    HookedStableDiffusionXLPipeline as HookedStableDiffusionPipeline,
                )
            else:
                from activation_collection.hooked_sd14_pipeline import (
                    HookedStableDiffusionPipeline,
                )

            self.pipe = HookedStableDiffusionPipeline.from_pretrained(
                self.cfg.model_name, torch_dtype=self.cfg.dtype, safety_checker=None
            )
            self.pipe.safety_checker = None
            self.pipe.to(self.accelerator.device)
            self.pipe.vae.to("cpu")
            self.pipe.set_progress_bar_config(disable=True)

            self.scheduler = self.pipe.scheduler
            self.scheduler.set_timesteps(self.cfg.num_inference_steps, device="cpu")
            self.scheduler_timesteps = self.scheduler.timesteps

            self.features_dict = {hookpoint: None for hookpoint in self.cfg.hook_names}

            dataset = load_dataset(
                self.cfg.dataset_name,
                split=self.cfg.split,
                columns=[self.cfg.column],
            )
            dataset = dataset.shuffle(self.cfg.seed)
            if limit := self.cfg.max_num_examples:
                dataset = dataset.select(range(limit))

            self.num_examples = len(dataset)
            dataloader = DataLoader(
                dataset,
                self.cfg.batch_size_per_gpu,
                shuffle=False,
                num_workers=self.cfg.num_workers,
            )
            self.dataloader = self.accelerator.prepare(dataloader)
            self.n_buffers = len(self.dataloader)

    @staticmethod
    def _consolidate_shards(
        source_dir: Path, output_dir: Path, copy_files: bool = True
    ) -> Dataset:
        """Consolidate sharded datasets into a single directory without rewriting data.

        Each of the shards must be of the same format, aka the full dataset must be able to
        be recreated like so:

        ```
        ds = concatenate_datasets(
            [Dataset.load_from_disk(str(shard_dir)) for shard_dir in sorted(source_dir.iterdir())]
        )
        ```

        Args:
            source_dir: Directory containing shard subdirectories (named shard_*)
            output_dir: Empty directory to consolidate into
            copy_files: If True, copy files; if False, move them and delete source_dir
        """
        assert source_dir.exists() and source_dir.is_dir()
        assert (
            output_dir.exists()
            and output_dir.is_dir()
            and not any(p for p in output_dir.iterdir() if p.name != ".tmp_shards")
        )

        shard_dirs = sorted(
            d for d in source_dir.iterdir() if d.name.startswith("shard_")
        )
        if not shard_dirs:
            raise Exception(f"No shards found in {source_dir}")

        transfer_fn = shutil.copy2 if copy_files else shutil.move

        transfer_fn(
            shard_dirs[0] / "dataset_info.json", output_dir / "dataset_info.json"
        )

        arrow_files = []
        file_count = 0

        for shard_dir in shard_dirs:
            state = json.loads((shard_dir / "state.json").read_text())
            for data_file in state["_data_files"]:
                src = shard_dir / data_file["filename"]
                new_name = f"data-{file_count:05d}-of-{len(shard_dirs):05d}.arrow"
                transfer_fn(src, output_dir / new_name)
                arrow_files.append({"filename": new_name})
                file_count += 1

        new_state = {
            "_data_files": arrow_files,
            "_fingerprint": None,
            "_format_columns": None,
            "_format_kwargs": {},
            "_format_type": None,
            "_output_all_columns": False,
            "_split": None,
        }

        with open(output_dir / "state.json", "w") as f:
            json.dump(new_state, f, indent=2)

        ds = Dataset.load_from_disk(str(output_dir))
        fingerprint = generate_fingerprint(ds)
        del ds

        with open(output_dir / "state.json", "r+") as f:
            state = json.loads(f.read())
            state["_fingerprint"] = fingerprint
            f.seek(0)
            json.dump(state, f, indent=2)
            f.truncate()

        if not copy_files:
            shutil.rmtree(source_dir)

        return Dataset.load_from_disk(str(output_dir))

    @torch.no_grad()
    def _create_shard(
        self,
        buffer: torch.Tensor,  # shape: (bs, n_steps, H*W, C)
        prompts: list[str],
        hook_name: str,
    ) -> Dataset:
        logger.info(f"{ buffer.shape= }")
        batch_size, n_steps, d_sample_size, d_in = buffer.shape
        buffer = buffer[:, :: self.cfg.cache_every_n_timesteps, :, :]
        n_steps = buffer.shape[1]

        activations = buffer.reshape(-1, d_sample_size, d_in)
        timesteps = self.scheduler_timesteps[
            :: self.cfg.cache_every_n_timesteps
        ].repeat(batch_size)
        repeated_prompts = [prompt for prompt in prompts for _ in range(n_steps)]

        return Dataset.from_dict(
            {
                "activations": activations,
                "timestep": timesteps,
                "prompt": repeated_prompts,
            },
            features=self.features_dict[hook_name],
        )

    def create_dataset_feature(self, hook_name: str, d_in: int, d_out: int) -> None:
        self.features_dict[hook_name] = Features(
            {
                "activations": Array2D(
                    shape=(d_in, d_out), dtype=TORCH_STRING_DTYPE_MAP[self.cfg.dtype]
                ),
                "timestep": Value(dtype="uint16"),
                "prompt": Value(dtype="string"),
            }
        )

    @torch.no_grad()
    def run(self) -> dict[str, Dataset]:
        assert self.cfg.new_cached_activations_path is not None
        rank = self.accelerator.process_index

        final_cached_activation_paths = {
            n: Path(os.path.join(self.cfg.new_cached_activations_path, n))
            for n in self.cfg.hook_names
        }

        if self.accelerator.is_main_process:
            for path in final_cached_activation_paths.values():
                path.mkdir(exist_ok=True, parents=True)
                if any(path.iterdir()):
                    raise Exception(
                        f"Activations directory ({path}) is not empty. Please delete it or specify a different path."
                    )
                (path / ".tmp_shards").mkdir()

        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            print(
                f"Started caching {self.num_examples} activations across {self.accelerator.num_processes} GPUs"
            )

        for i, batch in tqdm(
            enumerate(self.dataloader),
            desc=f"[rank {rank}] Caching activations",
            total=self.n_buffers,
            disable=not self.accelerator.is_main_process,
        ):
            try:
                prompt = batch[self.cfg.column]
                _, acts_cache = self.pipe.run_with_cache(
                    prompt=prompt,
                    device=self.accelerator.device,
                    output_type="latent",
                    num_inference_steps=self.cfg.num_inference_steps,
                    save_input=self.cfg.output_or_diff == "diff",
                    save_output=True,
                    positions_to_cache=self.cfg.hook_names,
                    guidance_scale=self.cfg.guidance_scale,
                )

                for hook_name in self.cfg.hook_names:
                    buffer = (
                        acts_cache["output"][hook_name] - acts_cache["input"][hook_name]
                        if self.cfg.output_or_diff == "diff"
                        else acts_cache["output"][hook_name]
                    )

                    if self.features_dict[hook_name] is None:
                        self.create_dataset_feature(
                            hook_name, buffer.shape[-2], buffer.shape[-1]
                        )

                    shard = self._create_shard(buffer, prompt, hook_name)
                    shard.save_to_disk(
                        f"{final_cached_activation_paths[hook_name]}/.tmp_shards/shard_{rank:02d}_{i:05d}",
                        num_shards=1,
                    )
                    del buffer, shard

            except StopIteration:
                print(f"[rank {rank}] Warning: ran out of samples at batch {i}")
                break

        self.accelerator.wait_for_everyone()

        datasets = {}
        if self.accelerator.is_main_process:
            for hook_name, path in final_cached_activation_paths.items():
                datasets[hook_name] = self._consolidate_shards(
                    path / ".tmp_shards", path, copy_files=False
                )
                print(f"Consolidated dataset for hook {hook_name}")

            if self.cfg.hf_repo_id:
                print("Pushing to hub...")
                for hook_name, dataset in datasets.items():
                    dataset.push_to_hub(
                        repo_id=f"{self.cfg.hf_repo_id}_{hook_name}",
                        num_shards=self.cfg.hf_num_shards or self.n_buffers,
                        private=self.cfg.hf_is_private_repo,
                        revision=self.cfg.hf_revision,
                    )

                meta_io = io.BytesIO()
                meta_io.write(_cfg_to_json(self.cfg).encode("utf-8"))
                meta_io.seek(0)

                HfApi().upload_file(
                    path_or_fileobj=meta_io,
                    path_in_repo="cache_activations_runner_cfg.json",
                    repo_id=self.cfg.hf_repo_id,
                    repo_type="dataset",
                    commit_message="Add cache_activations_runner metadata",
                )

        return datasets

    def load_and_push_to_hub(self) -> None:
        assert self.cfg.new_cached_activations_path is not None
        dataset = Dataset.load_from_disk(self.cfg.new_cached_activations_path)
        if self.accelerator.is_main_process:
            print("Loaded dataset from disk")
            if self.cfg.hf_repo_id:
                print("Pushing to hub...")
                dataset.push_to_hub(
                    repo_id=self.cfg.hf_repo_id,
                    num_shards=self.cfg.hf_num_shards
                    or (len(dataset) // self.cfg.batch_size_per_gpu),
                    private=self.cfg.hf_is_private_repo,
                    revision=self.cfg.hf_revision,
                )

                meta_io = io.BytesIO()
                meta_io.write(_cfg_to_json(self.cfg).encode("utf-8"))
                meta_io.seek(0)

                HfApi().upload_file(
                    path_or_fileobj=meta_io,
                    path_in_repo="cache_activations_runner_cfg.json",
                    repo_id=self.cfg.hf_repo_id,
                    repo_type="dataset",
                    commit_message="Add cache_activations_runner metadata",
                )
