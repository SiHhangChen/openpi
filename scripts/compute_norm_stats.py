"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import dataclasses
import os

import numpy as np
import torch
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class KeepStatsKeys(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        # Norm statistics are consumed only for these two model inputs. Dropping
        # images here avoids collating large camera tensors during the CPU pass.
        return {key: x[key] for key in ("state", "actions") if key in x}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    sampling_weights = getattr(dataset, "sampling_weights", None)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            KeepStatsKeys(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False

    # Keep normalization statistics aligned with joint-training sampling. The
    # weighted sampler draws from every source dataset with equal probability,
    # including when max_frames is used (rather than truncating at source 0).
    sampler = None
    if sampling_weights is not None and _data_loader.multi_sampling_mode() == "equal":
        generator = torch.Generator()
        generator.manual_seed(int(os.getenv("OPENPI_NORM_STATS_SEED", "0")))
        sampler = torch.utils.data.WeightedRandomSampler(
            torch.as_tensor(sampling_weights, dtype=torch.double),
            num_samples=num_batches * batch_size,
            replacement=True,
            generator=generator,
        )
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=int(os.getenv("OPENPI_NORM_STATS_NUM_WORKERS", num_workers)),
        shuffle=shuffle,
        sampler=sampler,
        num_batches=num_batches,
        framework="pytorch",
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            KeepStatsKeys(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(config_name: str, max_frames: int | None = None, assets_base_dir: str | None = None):
    config = _config.get_config(config_name)
    if assets_base_dir is not None:
        config = dataclasses.replace(config, assets_base_dir=assets_base_dir)
    data_config = config.data.create(config.assets_dirs, config.model)
    batch_size = int(os.getenv("OPENPI_NORM_STATS_BATCH_SIZE", config.batch_size))

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, batch_size, config.model, config.num_workers, max_frames
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = config.assets_dirs / (data_config.asset_id or data_config.repo_id)
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
