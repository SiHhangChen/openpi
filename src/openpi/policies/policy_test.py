from openpi_client import action_chunk_broker
import jax
import numpy as np
import pytest
import torch

from openpi.policies import aloha_policy
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


def test_reseed_restarts_jax_sampling_independently_of_previous_episodes():
    policy = object.__new__(_policy.Policy)
    policy._is_pytorch_model = False

    def sample_noise(seed):
        policy.reseed(seed)
        policy._rng, key = jax.random.split(policy._rng)
        return np.asarray(jax.random.normal(key, (50, 13)))

    first = sample_noise(1050)
    other = sample_noise(1051)
    resumed = sample_noise(1050)
    np.testing.assert_array_equal(first, resumed)
    assert not np.array_equal(first, other)


def test_reseed_restarts_pytorch_sampling():
    policy = object.__new__(_policy.Policy)
    policy._is_pytorch_model = True
    policy.reseed(1050)
    first = torch.randn(50, 13)
    policy.reseed(1051)
    other = torch.randn(50, 13)
    policy.reseed(1050)
    torch.testing.assert_close(first, torch.randn(50, 13), rtol=0, atol=0)
    assert not torch.equal(first, other)


@pytest.mark.manual
def test_infer():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    example = aloha_policy.make_aloha_example()
    result = policy.infer(example)

    assert result["actions"].shape == (config.model.action_horizon, 14)


@pytest.mark.manual
def test_broker():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    broker = action_chunk_broker.ActionChunkBroker(
        policy,
        # Only execute the first half of the chunk.
        action_horizon=config.model.action_horizon // 2,
    )

    example = aloha_policy.make_aloha_example()
    for _ in range(config.model.action_horizon):
        outputs = broker.infer(example)
        assert outputs["actions"].shape == (14,)
