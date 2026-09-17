import dataclasses

import numpy as np
import pytest

from openpi.models import pi0_config
from openpi.shared import normalize
from openpi.training import config as training_config
from openpi.training import data_loader

from . import compute_norm_stats


def test_stats_loader_uses_original_full_batch_policy(monkeypatch):
    length, batch_size = 5, 2
    dataset = [
        {"state": np.array([i], dtype=np.float32), "actions": np.array([[i]], dtype=np.float32)}
        for i in range(length)
    ]
    monkeypatch.setattr(data_loader, "create_torch_dataset", lambda *args: dataset)
    loader, num_batches = compute_norm_stats.create_torch_dataloader(
        training_config.DataConfig(repo_id="toy", sample_stride=1),
        action_horizon=1,
        batch_size=batch_size,
        model_config=pi0_config.Pi0Config(),
        num_workers=0,
    )

    batches = list(loader)
    assert num_batches == len(batches) == length // batch_size
    expected = np.arange(num_batches * batch_size)
    np.testing.assert_array_equal(np.concatenate([batch["state"] for batch in batches])[:, 0], expected)
    np.testing.assert_array_equal(
        np.concatenate([batch["actions"] for batch in batches])[:, 0, 0], expected
    )


def test_stats_uses_training_config_without_sampling_overrides(monkeypatch, tmp_path):
    config = training_config.get_config("pi05_membench_wr01_3view_full_stride10_10k")
    config = dataclasses.replace(config, assets_base_dir=str(tmp_path), batch_size=8, num_workers=0)
    expected_stride = config.data.sample_stride
    expected_horizon = config.model.action_horizon

    monkeypatch.setattr(training_config, "get_config", lambda name: config)
    monkeypatch.setattr(
        training_config.LeRobotMemBenchDataConfig,
        "create",
        lambda self, *args: training_config.DataConfig(repo_id=self.repo_id, sample_stride=self.sample_stride),
    )
    monkeypatch.setenv("OPENPI_NORM_STATS_NUM_WORKERS", "999")
    monkeypatch.setenv("OPENPI_NORM_STATS_BATCH_SIZE", "999")

    def create_dataset(data_config, action_horizon, model_config):
        assert data_config.sample_stride == expected_stride
        assert action_horizon == expected_horizon
        return [
            {
                "state": np.array([i], dtype=np.float32),
                "actions": np.full((action_horizon, 1), i, dtype=np.float32),
            }
            for i in range(0, 101, data_config.sample_stride)
        ]

    monkeypatch.setattr(data_loader, "create_torch_dataset", create_dataset)
    compute_norm_stats.main(config.name)

    assert (config.data.sample_stride, config.model.action_horizon) == (10, 50)
    stats = normalize.load(config.assets_dirs / "wr01-final")
    frames = np.arange(0, 101, expected_stride, dtype=np.float32)[:8]
    for key in ("state", "actions"):
        np.testing.assert_allclose(stats[key].mean, [frames.mean()])
        np.testing.assert_allclose(stats[key].std, [frames.std()], rtol=1e-5)


@pytest.mark.parametrize("parameter", ["sample_stride", "action_horizon", "assets_base_dir"])
def test_stats_does_not_accept_custom_sampling_parameters(parameter):
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        compute_norm_stats.main("debug", **{parameter: 1})


def test_remove_strings_preserves_numeric_inputs():
    inputs = {"state": np.array([1.0]), "actions": np.array([[2.0]]), "prompt": "instruction"}
    result = compute_norm_stats.RemoveStrings()(inputs)
    assert set(result) == {"state", "actions"}


def test_official_quantiles_are_not_expanded_for_sparse_actions():
    actions = np.zeros((10_000, 1), dtype=np.float32)
    actions[-1, 0] = 1.0
    stats = normalize.RunningStats()
    stats.update(actions)
    result = stats.get_statistics()
    assert abs(result.q01[0]) < 0.001
    assert abs(result.q99[0]) < 0.001
