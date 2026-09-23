from openpi.training.config import get_config


def test_real_robot_wr07_subgoal_config_uses_three_color_prompt_sidecars() -> None:
    config = get_config("pi05_real_robot_wr07_v3_subgoal")
    data = config.data
    mapping = data.repack_transforms.inputs[0].structure

    assert config.model.pi05
    assert (config.model.action_dim, config.model.action_horizon) == (32, 50)
    assert data.repo_id == "wr07_lerobot_v3_subtasks"
    assert data.assets.asset_id == "wr07_lerobot_v3"
    assert (data.state_keep_dim, data.action_output_dim, data.sample_stride) == (17, 17, 10)
    assert data.base_config.prompt_from_subtask
    assert not data.base_config.prompt_from_task
    assert mapping["prompt"] == "prompt"
    assert mapping["images"] == {
        "agentview_left": "observation.images.front",
        "agentview_right": "observation.images.left",
        "eye_in_hand": "observation.images.right",
    }
    assert (config.batch_size, config.num_train_steps) == (48, 30_000)
